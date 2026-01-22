from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
def CreatePasswordPolicyFromArgs(sql_messages, password_policy, args):
    """Generates password policy for the user.

  Args:
    sql_messages: module, The messages module that should be used.
    password_policy: sql_messages.UserPasswordValidationPolicy,
    The policy to build the new policy off.
    args: argparse.Namespace, The arguments that this command was invoked with.

  Returns:
    sql_messages.UserPasswordValidationPolicy or None

  """
    clear_password_policy = None
    if hasattr(args, 'clear_password_policy'):
        clear_password_policy = args.clear_password_policy
    allowed_failed_attempts = args.password_policy_allowed_failed_attempts
    password_expiration_duration = args.password_policy_password_expiration_duration
    enable_failed_attempts_check = args.password_policy_enable_failed_attempts_check
    enable_password_verification = args.password_policy_enable_password_verification
    should_generate_policy = any([allowed_failed_attempts is not None, password_expiration_duration is not None, enable_failed_attempts_check is not None, enable_password_verification is not None, clear_password_policy is not None])
    if not should_generate_policy:
        return None
    if password_policy is None:
        password_policy = sql_messages.UserPasswordValidationPolicy()
    if clear_password_policy:
        return sql_messages.UserPasswordValidationPolicy()
    if allowed_failed_attempts is not None:
        password_policy.allowedFailedAttempts = allowed_failed_attempts
        password_policy.enableFailedAttemptsCheck = True
    if password_expiration_duration is not None:
        password_policy.passwordExpirationDuration = str(password_expiration_duration) + 's'
    if enable_failed_attempts_check is not None:
        password_policy.enableFailedAttemptsCheck = enable_failed_attempts_check
    if enable_password_verification is not None:
        password_policy.enablePasswordVerification = enable_password_verification
    return password_policy