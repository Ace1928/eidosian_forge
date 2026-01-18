from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
Creates the set password request to send to UpdateUser.

  Args:
    sql_messages: module, The messages module that should be used.
    args: argparse.Namespace, The arguments that this command was invoked with.
    dual_password_type: How we want to interact with the dual password.
    project: the project that this user is in

  Returns:
    CreateSetPasswordRequest
  