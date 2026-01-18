from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import matrix_creator_common
from googlecloudsdk.api_lib.firebase.test import matrix_ops
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
Invoke the Testing service to create a test matrix from the user's args.

    Args:
      request_id: {str} a unique ID for the CreateTestMatrixRequest.

    Returns:
      The TestMatrix response message from the TestMatrices.Create rpc.

    Raises:
      HttpException if the test service reports an HttpError.
    