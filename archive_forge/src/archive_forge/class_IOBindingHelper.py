import logging
import traceback
from typing import TYPE_CHECKING
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_inference_collection import OrtValue
from onnxruntime.transformers.io_binding_helper import TypeHelper as ORTTypeHelper
from ..utils import is_cupy_available, is_onnxruntime_training_available
class IOBindingHelper:
    """
    A helper class to enable `ORTModel` instances to prepare IO binding  with dynamic shaped outputs for an inference session and transfer
    tensors from ONNX Runtime to other frameworks on device. It helps reduce memory copy between the host and device.
    """

    def __init__(self, model: ort.InferenceSession, device, **kwargs):
        self.model = model
        self.device = device
        self.model_inputs = {output_key.name: idx for idx, output_key in enumerate(model.get_inputs())}
        self.model_outputs = {output_key.name: idx for idx, output_key in enumerate(model.get_outputs())}
        self.model_input_names = list(self.model_inputs.keys())
        self.model_output_names = list(self.model_outputs.keys())

    @staticmethod
    def to_pytorch(ort_value: OrtValue) -> torch.Tensor:
        """
        Converts tensors held by OrtValues in ONNX runtime memory buffer to torch tensor.
        """
        if is_onnxruntime_training_available():
            return IOBindingHelper.to_pytorch_via_dlpack(ort_value)
        else:
            try:
                return IOBindingHelper.to_pytorch_via_cupy(ort_value)
            except Exception:
                logging.error(traceback.format_exc())
                logging.info('Unable to access output memory in CUDA, will offload to CPU')
                return IOBindingHelper.to_pytorch_via_numpy(ort_value)

    @staticmethod
    def to_pytorch_via_numpy(ort_value: OrtValue) -> torch.Tensor:
        ort_device = ort_value.device_name().lower()
        return torch.tensor(ort_value.numpy()).to(ort_device)

    @staticmethod
    def to_pytorch_via_cupy(ort_value: OrtValue) -> torch.Tensor:
        ort_device = ort_value.device_name().lower()
        if ort_device != 'cuda':
            raise RuntimeError(f'Exchange tensors to PyTorch via CuPy only when device is CUDA, got: {ort_device}')
        ort_type = ort_value.data_type()
        numpy_type = TypeHelper.ort_type_to_numpy_type(ort_type)
        memory = cp.cuda.UnownedMemory(ort_value.data_ptr(), 0, None)
        memory_ptr = cp.cuda.MemoryPointer(memory, 0)
        cp_array = cp.ndarray(shape=ort_value.shape(), memptr=memory_ptr, dtype=numpy_type)
        torch_tensor = torch.from_dlpack(cp_array.toDlpack())
        if 'bool' in ort_type:
            torch_tensor = torch_tensor.to(torch.bool)
        torch_tensor = torch_tensor.clone()
        return torch_tensor

    @staticmethod
    def to_pytorch_via_dlpack(ort_value: OrtValue) -> torch.Tensor:
        from torch._C import _from_dlpack
        torch_tensor = _from_dlpack(ort_value.to_dlpack())
        return torch_tensor

    @staticmethod
    def get_device_index(device):
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            return device
        return 0 if device.index is None else device.index

    @staticmethod
    def prepare_io_binding(ort_model: 'ORTModel', **inputs) -> ort.IOBinding:
        """
        Returns an IOBinding object for an inference session. This method is for general purpose, if the inputs and outputs
        are determined, you can prepare data buffers directly to avoid tensor transfers across frameworks.
        """
        if not all((input_name in inputs.keys() for input_name in ort_model.inputs_names)):
            raise ValueError(f'The ONNX model takes {ort_model.inputs_names.keys()} as inputs, but only {inputs.keys()} are given.')
        name_to_np_type = TypeHelper.get_io_numpy_type_map(ort_model.model)
        io_binding = ort_model.model.io_binding()
        for input_name in ort_model.inputs_names:
            onnx_input = inputs.pop(input_name)
            onnx_input = onnx_input.contiguous()
            io_binding.bind_input(input_name, onnx_input.device.type, ort_model.device.index, name_to_np_type[input_name], list(onnx_input.size()), onnx_input.data_ptr())
        for name in ort_model.output_names:
            io_binding.bind_output(name, ort_model.device.type, device_id=ort_model.device.index)
        return io_binding