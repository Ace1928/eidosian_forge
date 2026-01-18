import tokenize
from IPython.testing import tools as tt
from IPython.core import inputtransformer as ipt
def transform_and_reset(transformer):
    transformer = transformer()

    def transform(inp):
        try:
            return transformer.push(inp)
        finally:
            transformer.reset()
    return transform