from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import exceptions
import six
def _SplitStep(user_name):
    """Given a user name for a step, split it into the individual pieces.

  Examples:
     _SplitStep('Transform/Step') = ['Transform', 'Step']
     _SplitStep('Read(gs://Foo)/Bar') = ['Read(gs://Foo)', 'Bar']

  Args:
    user_name: The full user_name of the step.
  Returns:
    A list representing the individual pieces of the step name.
  """
    parens = 0
    accum = []
    step_parts = []
    for piece in user_name.split('/'):
        parens += piece.count('(') - piece.count(')')
        accum.append(piece)
        if parens == 0:
            step_parts.append(''.join(accum))
            del accum[:]
        else:
            accum.append('/')
    if accum:
        step_parts.append(accum)
    return step_parts