import re
from torch.nn import Module
from ..model import wav2vec2_model, Wav2Vec2Model
def _import_wav2vec2_pretraining(original: Module) -> Wav2Vec2Model:
    config = _parse_config(original)
    model = wav2vec2_model(**config, aux_num_out=None)
    model.load_state_dict(_convert_state_dict(original.state_dict()), strict=False)
    return model