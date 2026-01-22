from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import random
import string
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core.util import times
Generate a certificate id with the date and two length 3 alphanum strings.

  E.G. YYYYMMDD-ABC-DEF.

  Returns:
    The generated certificate id string.
  