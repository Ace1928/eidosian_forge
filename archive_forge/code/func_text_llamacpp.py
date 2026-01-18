from functools import singledispatch
from outlines.fsm.guide import StopAtEOSGuide
from outlines.generate import SequenceGenerator
from outlines.models import LlamaCpp, OpenAI
from outlines.models.llamacpp import LlamaSequenceGenerator
from outlines.samplers import Sampler, multinomial
@text.register(LlamaCpp)
def text_llamacpp(model: LlamaCpp, sampler: Sampler=multinomial()):
    if not isinstance(sampler, multinomial):
        raise NotImplementedError('The llama.cpp API does not support any other sampling algorithm ' + 'than the multinomial sampler.')
    generator = LlamaSequenceGenerator(None, model)
    return generator