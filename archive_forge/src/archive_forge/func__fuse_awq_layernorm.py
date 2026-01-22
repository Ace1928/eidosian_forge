from ..activations import ACT2FN
from ..modeling_utils import PreTrainedModel
from ..utils import is_auto_awq_available, is_torch_available
from ..utils.quantization_config import AwqBackendPackingMethod, AwqConfig, AWQLinearVersion
def _fuse_awq_layernorm(fuse_module_names, module, target_cls):
    """
    Fuse the LayerNorm layers into a target class using autoawq

    Args:
        fuse_module_names (`List[str]`):
            The list of module names to fuse
        module (`nn.Module`):
            The pytorch parent module that has layernorm modules to fuse
        target_cls (`~autoawq.FasterTransformerRMSNorm`):
            The `FasterTransformerRMSNorm` class as it only supports that class
            for now.
    """
    for module_name in fuse_module_names:
        if hasattr(module, module_name):
            old_module = getattr(module, module_name)
            module._modules[module_name] = target_cls(old_module.weight, old_module.variance_epsilon).to(old_module.weight.device)
            del old_module