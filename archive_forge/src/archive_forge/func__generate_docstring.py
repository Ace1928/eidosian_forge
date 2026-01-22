import warnings
from typing import Any, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch import Tensor
from torch.masked import as_masked_tensor, is_masked_tensor, MaskedTensor
from . import _docs
from torch._prims_common import corresponding_real_dtype
from torch import sym_float
def _generate_docstring(func):
    """A utility function called from tools/update_masked_docs.py
    script to update the module torch.masked._docs.py
    """
    docstring_templates = dict(reduction_signature='{function_name}(input, {operation_args}, *, {operation_kwargs}) -> Tensor', reduction_descr='Returns {operation name} of all the elements in the :attr:`input`\ntensor along the given dimension(s) :attr:`dim` while the :attr:`input`\nelements are masked out according to the boolean tensor\n:attr:`mask`.', reduction_args='If :attr:`keepdim` is ``True``, the output tensor is of the same size\nas :attr:`input` except in the dimension(s) :attr:`dim` where it is of\nsize 1. Otherwise, :attr:`dim` is squeezed (see\n:func:`torch.squeeze`), resulting in the output tensor having 1 (or\n``len(dim)``) fewer dimension(s).\n\nThe boolean tensor :attr:`mask` defines the "validity" of\n:attr:`input` tensor elements: if :attr:`mask` element is True\nthen the corresponding element in :attr:`input` tensor will be\nincluded in {operation name} computation, otherwise the element is\nignored.\n\nWhen all elements of :attr:`input` along the given dimension\n:attr:`dim` are ignored (fully masked-out), the corresponding element\nof the output tensor will have undefined value: it may or may not\ncorrespond to the identity value of {operation name} operation; the\nchoice may correspond to the value that leads to the most efficient\nstorage of :attr:`output` tensor.\n\nThe mask of the output tensor can be computed as\n``torch.any(torch.broadcast_to(mask, input.shape), dim, keepdim=keepdim,\ndtype=torch.bool)``.\n\nThe shapes of the :attr:`mask` tensor and the :attr:`input` tensor\ndon\'t need to match, but they must be :ref:`broadcastable\n<broadcasting-semantics>` and the dimensionality of the :attr:`mask`\ntensor must not be greater than of the :attr:`input` tensor.\n\nArgs:\n    input (Tensor): the input tensor\n    {args_declarations}\n\nKeyword args:\n    {kwargs_declarations}', reduction_example='Example::\n\n    >>> input = {example_input}\n    >>> input\n    {indent_example_input}\n    >>> mask = {example_mask}\n    >>> mask\n    {indent_example_mask}\n    >>> {full_function_name}(input, {example_args}, mask=mask)\n    {indent_example_output}\n', reduction_identity='The identity value of {operation name} operation, which is used to start the reduction, is ``{identity_int32}``.', reduction_identity_dtype='The identity value of {operation name} operation, which is used to start the\nreduction, depends on input dtype. For instance, for float32, uint8,\nand int32 dtypes, the identity values are ``{identity_float32}``, ``{identity_uint8}``, and ``{identity_int32}``, respectively.', normalization_signature='{function_name}(input, {operation_args}, *, {operation_kwargs}) -> Tensor', normalization_descr='Returns {operation name} of all the slices in the :attr:`input` tensor\nalong :attr:`dim` while the :attr:`input` elements are masked out\naccording to the boolean tensor :attr:`mask`.\n\n{definition}', normalization_args='The boolean tensor :attr:`mask` defines the "validity" of\n:attr:`input` tensor elements: if :attr:`mask` element is True then\nthe corresponding element in :attr:`input` tensor will be included in\n{operation name} computation, otherwise the element is ignored.\n\nThe values of masked-out elements of the output tensor have undefined\nvalue: it may or may not be set to zero or nan; the choice may correspond to\nthe value that leads to the most efficient storage of :attr:`output`\ntensor.\n\nThe mask of the {operation name} output tensor can be computed as\n``torch.broadcast_to(mask, input.shape)``.\n\nThe shapes of the :attr:`mask` tensor and the :attr:`input` tensor\ndon\'t need to match, but they must be :ref:`broadcastable\n<broadcasting-semantics>` and the dimensionality of the :attr:`mask`\ntensor must not be greater than of the :attr:`input` tensor.\n\nArgs:\n    input (Tensor): the input tensor\n    {args_declarations}\n\nKeyword args:\n    {kwargs_declarations}', normalization_example='Example::\n\n    >>> input = {example_input}\n    >>> input\n    {indent_example_input}\n    >>> mask = {example_mask}\n    >>> mask\n    {indent_example_mask}\n    >>> {full_function_name}(input, {example_args}, mask=mask)\n    {indent_example_output}\n')
    args_and_kwargs = dict(sum=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')), prod=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')), cumsum=(('dim__as_int',), ('dtype=None', 'mask=None')), cumprod=(('dim__as_int',), ('dtype=None', 'mask=None')), amin=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')), amax=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')), argmin=(('dim__as_int',), ('keepdim=False', 'dtype=None', 'mask=None')), argmax=(('dim__as_int',), ('keepdim=False', 'dtype=None', 'mask=None')), mean=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')), median=(('dim__as_int',), ('keepdim=False', 'dtype=None', 'mask=None')), norm=(('ord', 'dim'), ('keepdim=False', 'dtype=None', 'mask=None')), var=(('dim', 'unbiased'), ('keepdim=False', 'dtype=None', 'mask=None')), std=(('dim', 'unbiased'), ('keepdim=False', 'dtype=None', 'mask=None')), logsumexp=(('dim',), ('keepdim=False', 'dtype=None', 'mask=None')), softmax=(('dim__as_int',), ('dtype=None', 'mask=None')), log_softmax=(('dim__as_int',), ('dtype=None', 'mask=None')), softmin=(('dim__as_int',), ('dtype=None', 'mask=None')), normalize=(('ord__required', 'dim__as_int'), ('eps=1e-12', 'dtype=None', 'mask=None')))
    argument_declarations = dict(dim='dim (int or tuple of ints, optional): the dimension or dimensions to reduce.\n  Default: None that is equivalent to ``tuple(range(input.ndim))``.', dim__as_int='dim (int): the dimension along which {operation name} is computed.', ord='ord (int, float, optional): the order of vector norm. Default: 2.\n  See :func:`torch.linalg.vector_norm` for a list of supported norms.', ord__required='ord (int, float): the order of vector norm. Default: 2.\n  See :func:`torch.linalg.vector_norm` for a list of supported norms.', unbiased='unbiased (bool): when True, use Bessel’s correction, otherwise, compute\n  the uncorrected sample variance.', eps='eps (float, optional): small value to avoid division by zero. Default: {default}.', keepdim='keepdim (bool, optional): whether the output tensor has\n  :attr:`dim` retained or not. Default: {default}.', dtype='dtype (:class:`torch.dtype`, optional): the desired data type\n  of returned tensor.  If specified, the input tensor is\n  casted to :attr:`dtype` before the operation is\n  performed. Default: {default}.', mask='mask (:class:`torch.Tensor`, optional): the boolean tensor\n  containing the binary mask of validity of input tensor\n  elements.\n  Default: None that is equivalent to ``torch.ones(input.shape, dtype=torch.bool)``.')
    definitions = dict(softmax='Let ``x`` be a sequence of unmasked elements of one-dimensional slice\nof the :attr:`input` tensor. Softmax of i-th element in ``x`` is\ndefined as ``exp(x[i])/sum(exp(x))``.', log_softmax='Let ``x`` be a sequence of unmasked elements of one-dimensional slice\nof the :attr:`input` tensor. LogSoftmax of i-th element in ``x`` is\ndefined as ``log(exp(x[i])/sum(exp(x)))``.', softmin='Let ``x`` be a sequence of unmasked elements of one-dimensional slice\nof the :attr:`input` tensor. Softmin of i-th element in ``x`` is\ndefined as ``exp(-x[i])/sum(exp(-x))``.', normalize='Let ``x`` be a sequence of unmasked elements of one-dimensional slice\nof the :attr:`input` tensor. Normalize of i-th element in ``x`` is\ndefined as ``x[i]/max(norm(x, p), eps)``.', cumsum='Let ``x`` be a sequence of unmasked elements of one-dimensional slice\nof the :attr:`input` tensor. Cumsum of i-th element in ``x`` is\ndefined as ``sum(x[:i])``.', cumprod='Let ``x`` be a sequence of unmasked elements of one-dimensional slice\nof the :attr:`input` tensor. Cumsum of i-th element in ``x`` is\ndefined as ``prod(x[:i])``.')
    reduction_names = dict(sum='sum', prod='product', amax='maximum', amin='minimum', argmax='argmax', argmin='argmin', mean='mean', median='median', norm='norm', var='variance', std='standard_deviation', logsumexp='logsumexp')
    normalization_names = dict(softmax='softmax', log_softmax='log_softmax', softmin='softmin', normalize='normalize', cumsum='cumulative_sum', cumprod='cumulative_prod')
    operation_names = {}
    operation_names.update(reduction_names)
    operation_names.update(normalization_names)
    example_dim = 1
    example_input = torch.tensor([[-3, -2, -1], [0, 1, 2]])
    example_mask = torch.tensor([[True, False, True], [False, False, False]])
    example_args: Tuple[Any, ...]
    if func.__name__ in {'norm', 'normalize'}:
        example_args = (2.0, example_dim)
        example_input = example_input.to(dtype=torch.float32)
    elif func.__name__ in {'var', 'std'}:
        example_args = (example_dim, False)
    elif func.__name__ == 'median':
        example_args = (example_dim,)
        example_input = example_input.to(dtype=torch.float32)
    else:
        example_args = (example_dim,)
    operation_args: Tuple[str, ...]
    operation_kwargs: Tuple[str, ...]
    operation_args, operation_kwargs = args_and_kwargs[func.__name__]
    arg_declarations = ['\n    '.join(argument_declarations.get(a, f'{a.split('__', 1)[0]}: TBD.').splitlines()) for a in operation_args]
    kwarg_declarations = ['\n    '.join(argument_declarations.get(a.split('=', 1)[0], f'{a.split('__', 1)[0]}: TBD.').format(default=a.split('=', 1)[1]).splitlines()) for a in operation_kwargs]
    if func.__name__ in reduction_names:
        op_kind = 'reduction'
        doc_sections = ['signature', 'descr', 'identity', 'args', 'example']
    elif func.__name__ in normalization_names:
        op_kind = 'normalization'
        doc_sections = ['signature', 'descr', 'args', 'example']
        example_input = example_input.to(dtype=torch.float32)
    else:
        assert 0
    example_output = func(example_input, *example_args, mask=example_mask)
    template_data = {'function_name': func.__name__, 'full_function_name': func.__module__ + '.' + func.__name__, 'operation name': operation_names[func.__name__], 'operation_args': ', '.join((a.split('__', 1)[0] for a in operation_args)), 'operation_kwargs': ', '.join((a.split('__', 1)[0] for a in operation_kwargs)), 'example_input': ' '.join(str(example_input).split()), 'example_args': ', '.join(map(str, example_args)), 'example_mask': ' '.join(str(example_mask).split()), 'indent_example_input': '\n    '.join(str(example_input).splitlines()), 'indent_example_mask': '\n    '.join(str(example_mask).splitlines()), 'indent_example_output': '\n    '.join(str(example_output).splitlines())}
    if func.__name__ in reduction_names:
        template_data.update(identity_uint8=_reduction_identity(func.__name__, torch.tensor(0, dtype=torch.uint8)), identity_int32=_reduction_identity(func.__name__, torch.tensor(0, dtype=torch.int32)), identity_float32=_reduction_identity(func.__name__, torch.tensor(0, dtype=torch.float32)))
        if func.__name__ == 'norm':
            template_data.update(identity_ord_ninf=_reduction_identity(func.__name__, torch.tensor(0, dtype=torch.float32), float('-inf')))
    elif func.__name__ in normalization_names:
        template_data.update(definition=definitions[func.__name__])
    else:
        assert 0
    template_data.update(args_declarations='\n    '.join(arg_declarations).format_map(template_data))
    template_data.update(kwargs_declarations='\n    '.join(kwarg_declarations).format_map(template_data))
    templates = {k: v.format_map(template_data) for k, v in docstring_templates.items() if k.startswith(op_kind)}
    templates.update(((k, v.format_map(template_data) if isinstance(v, str) else v) for k, v in template_data.items()))
    if func.__doc__ is None:
        doc_template = '\n\n'.join([f'{{{op_kind}_{sec}}}' for sec in doc_sections])
    else:
        doc_template = func.__doc__
    return doc_template.format_map(templates)