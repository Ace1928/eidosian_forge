from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import re
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.core import log
from  googlecloudsdk.core.util.files import FileReader
def _GetJustifications(vuln, product, msgs):
    """Get justifications.

  Args:
    vuln: vulnerability proto
    product: product proto
    msgs: container analysis messages

  Returns:
    justification proto
  """
    justification_type_as_string = 'justification_type_unspecified'
    justification_type = None
    flags = vuln.get('flags')
    if flags is None:
        return msgs.Justification()
    for flag in flags:
        label = flag.get('label')
        for product_id in flag.get('product_ids'):
            if product_id == product.id:
                justification_type_as_string = label
    enum_dict = msgs.Justification.JustificationTypeValueValuesEnum.to_dict()
    number = enum_dict[justification_type_as_string.upper()]
    justification_type = msgs.Justification.JustificationTypeValueValuesEnum(number)
    justification = msgs.Justification(justificationType=justification_type)
    return justification