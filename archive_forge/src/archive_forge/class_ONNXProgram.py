from __future__ import (  # for onnx.ModelProto (ONNXProgram) and onnxruntime (ONNXRuntimeOptions)
import abc
import contextlib
import dataclasses
import io
import logging
import os
import warnings
from collections import defaultdict
from typing import (
from typing_extensions import Self
import torch
import torch._ops
import torch.export as torch_export
import torch.utils._pytree as pytree
from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, io_adapter
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.fx import (
class ONNXProgram:
    """An in-memory representation of a PyTorch model that has been exported to ONNX.

    Args:
        model_proto: The exported ONNX model as an :py:obj:`onnx.ModelProto`.
        input_adapter: The input adapter used to convert PyTorch inputs into ONNX inputs.
        output_adapter: The output adapter used to convert PyTorch outputs into ONNX outputs.
        diagnostic_context: Context object for the SARIF diagnostic system responsible for logging errors and metadata.
        fake_context: The fake context used for symbolic tracing.
        export_exception: The exception that occurred during export, if any.
        model_signature: The model signature for the exported ONNX graph.
    """
    _model_proto: Final[onnx.ModelProto]
    _input_adapter: Final[io_adapter.InputAdapter]
    _output_adapter: Final[io_adapter.OutputAdapter]
    _diagnostic_context: Final[diagnostics.DiagnosticContext]
    _fake_context: Final[Optional[ONNXFakeContext]]
    _export_exception: Final[Optional[Exception]]
    _model_signature: Final[Optional[torch.export.ExportGraphSignature]]
    _model_torch: Final[Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]]

    @_beartype.beartype
    def __init__(self, model_proto: onnx.ModelProto, input_adapter: io_adapter.InputAdapter, output_adapter: io_adapter.OutputAdapter, diagnostic_context: diagnostics.DiagnosticContext, *, fake_context: Optional[ONNXFakeContext]=None, export_exception: Optional[Exception]=None, model_signature: Optional[torch.export.ExportGraphSignature]=None, model_torch: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None):
        self._model_proto = model_proto
        self._model_signature = model_signature
        self._model_torch = model_torch
        self._input_adapter = input_adapter
        self._output_adapter = output_adapter
        self._diagnostic_context = diagnostic_context
        self._fake_context = fake_context
        self._export_exception = export_exception

    def __call__(self, *args: Any, model_with_state_dict: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None, options: Optional[ONNXRuntimeOptions]=None, **kwargs: Any) -> Any:
        """Runs the ONNX model using ONNX Runtime

        Args:
            args: The positional inputs to the model.
            kwargs: The keyword inputs to the model.
            model_with_state_dict: The PyTorch model to fetch state from.
                Required when :func:`enable_fake_mode` is used to extract real initializers as needed by the ONNX graph.
            options: The options to use for running the model with ONNX Runtime.

        Returns:
            The model output as computed by ONNX Runtime
        """
        import onnxruntime
        model_with_state_dict = model_with_state_dict or self._model_torch
        onnx_input = self.adapt_torch_inputs_to_onnx(*args, model_with_state_dict=model_with_state_dict, **kwargs)
        options = options or ONNXRuntimeOptions()
        providers = options.execution_providers or onnxruntime.get_available_providers()
        onnx_model = self.model_proto.SerializeToString()
        ort_session = onnxruntime.InferenceSession(onnx_model, providers=providers)
        onnxruntime_input = {k.name: v.numpy(force=True) for k, v in zip(ort_session.get_inputs(), onnx_input)}
        return ort_session.run(None, onnxruntime_input)

    @property
    def model_proto(self) -> onnx.ModelProto:
        """The exported ONNX model as an :py:obj:`onnx.ModelProto`."""
        if self._export_exception is not None:
            raise self._export_exception
        return self._model_proto

    @property
    def model_signature(self) -> Optional[torch.export.ExportGraphSignature]:
        """The model signature for the exported ONNX graph.

        This information is relevant because ONNX specification often differs from PyTorch's, resulting
        in a ONNX graph with input and output schema different from the actual PyTorch model implementation.
        By using the model signature, the users can understand the inputs and outputs differences
        and properly execute the model in ONNX Runtime.

        NOTE: Model signature is only available when the ONNX graph was exported from a
        :class:`torch.export.ExportedProgram` object.

        NOTE: Any transformation done to the model that changes the model signature must be accompanied
        by updates to this model signature as well through :class:`InputAdaptStep` and/or :class:`OutputAdaptStep`.

        Example:

            The following model produces different sets of inputs and outputs.
            The first 4 inputs are model parameters (namely conv1.weight, conv2.weight, fc1.weight, fc2.weight),
            and the next 2 inputs are registered buffers (namely my_buffer2, my_buffer1) and finally
            the last 2 inputs are user inputs (namely x and b).
            The first output is a buffer mutation (namely my_buffer2) and the last output is the actual model output.

            >>> class CustomModule(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))
            ...         self.register_buffer("my_buffer1", torch.tensor(3.0))
            ...         self.register_buffer("my_buffer2", torch.tensor(4.0))
            ...         self.conv1 = torch.nn.Conv2d(1, 32, 3, 1, bias=False)
            ...         self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, bias=False)
            ...         self.fc1 = torch.nn.Linear(9216, 128, bias=False)
            ...         self.fc2 = torch.nn.Linear(128, 10, bias=False)
            ...     def forward(self, x, b):
            ...         tensor_x = self.conv1(x)
            ...         tensor_x = torch.nn.functional.sigmoid(tensor_x)
            ...         tensor_x = self.conv2(tensor_x)
            ...         tensor_x = torch.nn.functional.sigmoid(tensor_x)
            ...         tensor_x = torch.nn.functional.max_pool2d(tensor_x, 2)
            ...         tensor_x = torch.flatten(tensor_x, 1)
            ...         tensor_x = self.fc1(tensor_x)
            ...         tensor_x = torch.nn.functional.sigmoid(tensor_x)
            ...         tensor_x = self.fc2(tensor_x)
            ...         output = torch.nn.functional.log_softmax(tensor_x, dim=1)
            ...         (
            ...         self.my_buffer2.add_(1.0) + self.my_buffer1
            ...         )  # Mutate buffer through in-place addition
            ...         return output
            >>> inputs = (torch.rand((64, 1, 28, 28), dtype=torch.float32), torch.randn(3))
            >>> exported_program = torch.export.export(CustomModule(), args=inputs)
            >>> onnx_program = torch.onnx.dynamo_export(exported_program, *inputs)
            >>> print(onnx_program.model_signature)
            ExportGraphSignature(
                input_specs=[
                    InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg0_1'), target='conv1.weight'),
                    InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg1_1'), target='conv2.weight'),
                    InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg2_1'), target='fc1.weight'),
                    InputSpec(kind=<InputKind.PARAMETER: 2>, arg=TensorArgument(name='arg3_1'), target='fc2.weight'),
                    InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg4_1'), target='my_buffer2'),
                    InputSpec(kind=<InputKind.BUFFER: 3>, arg=TensorArgument(name='arg5_1'), target='my_buffer1'),
                    InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='l_x_'), target=None),
                    InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='arg1'), target=None)
                ],
                output_specs=[
                    OutputSpec(kind=<OutputKind.BUFFER_MUTATION: 3>, arg=TensorArgument(name='add'), target='my_buffer2'),
                    OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='_log_softmax'), target=None)
                ]
            )
        """
        return self._model_signature

    @property
    def diagnostic_context(self) -> diagnostics.DiagnosticContext:
        """The diagnostic context associated with the export."""
        return self._diagnostic_context

    @property
    def fake_context(self) -> Optional[ONNXFakeContext]:
        """The fake context associated with the export."""
        return self._fake_context

    @_beartype.beartype
    def adapt_torch_inputs_to_onnx(self, *model_args, model_with_state_dict: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None, **model_kwargs) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        """Converts the PyTorch model inputs to exported ONNX model inputs format.

        Due to design differences, input/output format between PyTorch model and exported
        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are
        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,
        but only flattened tensors are supported by ONNX, etc.

        The actual adapting steps are associated with each individual export. It
        depends on the PyTorch model, the particular set of model_args and model_kwargs
        used for the export, and export options.

        This method replays the adapting steps recorded during export.

        Args:
            model_args: The PyTorch model inputs.
            model_with_state_dict: The PyTorch model to get extra state from.
                If not specified, the model used during export is used.
                Required when :func:`enable_fake_mode` is used to extract real initializers as needed by the ONNX graph.
            model_kwargs: The PyTorch model keyword inputs.

        Returns:
            A sequence of tensors converted from PyTorch model inputs.

        Example::

            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> import torch
            >>> import torch.onnx
            >>> from typing import Dict, Tuple
            >>> def func_nested_input(
            ...     x_dict: Dict[str, torch.Tensor],
            ...     y_tuple: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            ... ):
            ...     if "a" in x_dict:
            ...         x = x_dict["a"]
            ...     elif "b" in x_dict:
            ...         x = x_dict["b"]
            ...     else:
            ...         x = torch.randn(3)
            ...
            ...     y1, (y2, y3) = y_tuple
            ...
            ...     return x + y1 + y2 + y3
            >>> x_dict = {"a": torch.tensor(1.)}
            >>> y_tuple = (torch.tensor(2.), (torch.tensor(3.), torch.tensor(4.)))
            >>> onnx_program = torch.onnx.dynamo_export(func_nested_input, x_dict, y_tuple)
            >>> print(x_dict, y_tuple)
            {'a': tensor(1.)} (tensor(2.), (tensor(3.), tensor(4.)))
            >>> print(onnx_program.adapt_torch_inputs_to_onnx(x_dict, y_tuple, model_with_state_dict=func_nested_input))
            (tensor(1.), tensor(2.), tensor(3.), tensor(4.))

        .. warning::
            This API is experimental and is *NOT* backward-compatible.

        """
        model_with_state_dict = model_with_state_dict or self._model_torch
        assert model_with_state_dict is not None, 'model_with_state_dict must be specified.'
        return self._input_adapter.apply(*model_args, model=model_with_state_dict, **model_kwargs)

    @_beartype.beartype
    def adapt_torch_outputs_to_onnx(self, model_outputs: Any, model_with_state_dict: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]]=None) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        """Converts the PyTorch model outputs to exported ONNX model outputs format.

        Due to design differences, input/output format between PyTorch model and exported
        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are
        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,
        but only flattened tensors are supported by ONNX, etc.

        The actual adapting steps are associated with each individual export. It
        depends on the PyTorch model, the particular set of model_args and model_kwargs
        used for the export, and export options.

        This method replays the adapting steps recorded during export.

        Args:
            model: The PyTorch model to get extra state from.
            model_outputs: The PyTorch model outputs.
            model_with_state_dict: The PyTorch model to get extra state from.
                If not specified, the model used during export is used.
                Required when :func:`enable_fake_mode` is used to extract real initializers as needed by the ONNX graph.

        Returns:
            PyTorch model outputs in exported ONNX model outputs format.

        Example::

            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> import torch
            >>> import torch.onnx
            >>> def func_returning_tuples(x, y, z):
            ...     x = x + y
            ...     y = y + z
            ...     z = x + y
            ...     return (x, (y, z))
            >>> x = torch.tensor(1.)
            >>> y = torch.tensor(2.)
            >>> z = torch.tensor(3.)
            >>> onnx_program = torch.onnx.dynamo_export(func_returning_tuples, x, y, z)
            >>> pt_output = func_returning_tuples(x, y, z)
            >>> print(pt_output)
            (tensor(3.), (tensor(5.), tensor(8.)))
            >>> print(onnx_program.adapt_torch_outputs_to_onnx(pt_output, model_with_state_dict=func_returning_tuples))
            [tensor(3.), tensor(5.), tensor(8.)]

        .. warning::
            This API is experimental and is *NOT* backward-compatible.

        """
        model_with_state_dict = model_with_state_dict or self._model_torch
        assert model_with_state_dict is not None, 'model_with_state_dict must be specified.'
        return self._output_adapter.apply(model_outputs, model=model_with_state_dict)

    @_beartype.beartype
    def save(self, destination: Union[str, io.BufferedIOBase], *, model_state_dict: Optional[Union[Dict[str, Any], str]]=None, serializer: Optional[ONNXProgramSerializer]=None) -> None:
        """Saves the in-memory ONNX model to ``destination`` using specified ``serializer``.

        Args:
            destination: The destination to save the ONNX model. It can be either a string or a file-like object.
                When used with ``model_state_dict``, it must be a string with a full path to the destination.
                In that case, besides saving the ONNX model, a folder with "_initializers" suffix (without extension)
                will be created to store the each initializer of the ONNX model in a separate file. For example, if the
                destination is "/path/model.onnx", the initializers will be saved in "/path/model_initializers/" folder.
            model_state_dict: The state_dict of the PyTorch model containing all weights on it.
                It can be either a dict as returned by :meth:`model.state_dict`, or a string with a file name.
                Required when :func:`enable_fake_mode` is used but real initializers are needed on the ONNX graph.
                It can be either a string with the path to a checkpoint or a dictionary with the actual model state.

            serializer: The serializer to use. If not specified, the model will be serialized as Protobuf.
        """
        if serializer is None:
            if isinstance(destination, str):
                serializer = LargeProtobufONNXProgramSerializer(destination)
            else:
                serializer = ProtobufONNXProgramSerializer()
        _model_state_dict_files: List[Union[str, io.BytesIO]] = []
        if model_state_dict is not None:
            if isinstance(model_state_dict, dict):
                model_state_dict_file = io.BytesIO()
                torch.save(model_state_dict, model_state_dict_file)
                model_state_dict_file.seek(0)
                _model_state_dict_files.append(model_state_dict_file)
            else:
                (isinstance(model_state_dict, str), "model_state_dict must be a path to the model's state_dict or the actual state_dict")
                _model_state_dict_files.append(model_state_dict)
        elif self._fake_context and self._fake_context.state_dict_paths:
            for path in self._fake_context.state_dict_paths:
                if path in _model_state_dict_files:
                    continue
                try:
                    extra_state_dict = torch.load(path)
                    extra_state_dict_file = io.BytesIO()
                    torch.save(extra_state_dict, extra_state_dict_file)
                    extra_state_dict_file.seek(0)
                    _model_state_dict_files.append(extra_state_dict_file)
                except FileNotFoundError:
                    pass
        if _model_state_dict_files:
            if not isinstance(destination, str):
                raise RuntimeError('`destination` must be a string with a path when `model_state_dict` is specified.')
            destination_path, destination_filename = os.path.split(destination)
            onnx_model_location = destination_filename
            onnx_initializer_location = destination_filename.split('.')[0] + '_initializers'
            fx_serialization.save_model_with_external_data(destination_path, onnx_model_location, onnx_initializer_location, tuple(_model_state_dict_files), self.model_proto)
        elif isinstance(destination, str):
            with open(destination, 'wb') as f:
                serializer.serialize(self, f)
        else:
            try:
                serializer.serialize(self, destination)
            except ValueError as exc:
                raise ValueError("'destination' should be provided as a path-like string when saving a model larger than 2GB. External tensor data will be saved alongside the model on disk.") from exc

    @_beartype.beartype
    def save_diagnostics(self, destination: str) -> None:
        """Saves the export diagnostics as a SARIF log to the specified destination path.

        Args:
            destination: The destination to save the diagnostics SARIF log.
                It must have a `.sarif` extension.

        Raises:
            ValueError: If the destination path does not end with `.sarif` extension.
        """
        if not destination.endswith('.sarif'):
            message = f"'destination' must have a .sarif extension, got {destination}"
            log.fatal(message)
            raise ValueError(message)
        self.diagnostic_context.dump(destination)

    @classmethod
    def _from_failure(cls, export_exception: Exception, diagnostic_context: diagnostics.DiagnosticContext) -> Self:
        """
        Creates an instance of :class:`ONNXProgram` when the export process encounters a failure.

        In case of a failed export, this method is used to encapsulate the exception
        and associated diagnostic context within an :class:`ONNXProgram` instance for
        easier handling and debugging.

        Args:
            export_exception: The exception raised during the export process.
            diagnostic_context: The context associated with diagnostics during export.

        Returns:
            An instance of :class:`ONNXProgram` representing the failed ONNX program.
        """
        import onnx
        return ONNXProgram(onnx.ModelProto(), io_adapter.InputAdapter(), io_adapter.OutputAdapter(), diagnostic_context, export_exception=export_exception)