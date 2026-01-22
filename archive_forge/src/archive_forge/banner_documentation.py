import os
import string
from taskflow.utils import misc
from taskflow import version
Makes a taskflow banner string.

    For example::

      >>> from taskflow.utils import banner
      >>> chapters = {
          'Connection details': {
              'Topic': 'hello',
          },
          'Powered by': {
              'Executor': 'parallel',
          },
      }
      >>> print(banner.make_banner('Worker', chapters))

    This will output::

      ___    __
       |    |_
       |ask |low v1.26.1
      *Worker*
      Connection details:
        Topic => hello
      Powered by:
        Executor => parallel
    