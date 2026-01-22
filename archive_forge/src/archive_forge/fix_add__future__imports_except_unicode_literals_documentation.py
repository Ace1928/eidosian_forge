from lib2to3 import fixer_base
from libfuturize.fixer_util import future_import

Fixer for adding:

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function

This is "stage 1": hopefully uncontroversial changes.

Stage 2 adds ``unicode_literals``.
