import ast
import inspect
import textwrap
import warnings
import torch
Determine if a Call node is 'torch.jit.annotate' in __init__.

        Visit a Call node in an ``nn.Module``'s ``__init__``
        method and determine if it's ``torch.jit.annotate``. If so,
        see if it conforms to our attribute annotation rules.
        