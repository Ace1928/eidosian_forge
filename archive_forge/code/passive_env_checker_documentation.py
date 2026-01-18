import inspect
from functools import partial
from typing import Callable
import numpy as np
from gym import Space, error, logger, spaces
A passive check of the `Env.render` that the declared render modes/fps in the metadata of the environment is declared.