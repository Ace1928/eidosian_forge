import logging
from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES
def _attach_model_to_data_sparsifier(module, data_sparsifier, config=None):
    """Attaches a data sparsifier to all the layers of the module.
    Essentially, loop over all the weight parameters in the module and
    attach it to the data sparsifier.
    Note::
        The '.' in the layer names are replaced with '_' (refer to _get_valid_name() below)
        before attaching to the sparsifier. This is because, the data
        sparsifier uses a dummy model inside to store the weight parameters.
    """
    if config is None:
        config = {}
    for name, parameter in module.named_parameters():
        if type(parameter) in SUPPORTED_TYPES:
            valid_name = _get_valid_name(name)
            data_sparsifier.add_data(name=valid_name, data=parameter, **config.get(valid_name, {}))