import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def update_account_password_policy(self, allow_users_to_change_password=None, hard_expiry=None, max_password_age=None, minimum_password_length=None, password_reuse_prevention=None, require_lowercase_characters=None, require_numbers=None, require_symbols=None, require_uppercase_characters=None):
    """
        Update the password policy for the AWS account.

        Notes: unset parameters will be reset to Amazon default settings!
            Most of the password policy settings are enforced the next time your users
            change their passwords. When you set minimum length and character type
            requirements, they are enforced the next time your users change their
            passwords - users are not forced to change their existing passwords, even
            if the pre-existing passwords do not adhere to the updated password
            policy. When you set a password expiration period, the expiration period
            is enforced immediately.

        :type allow_users_to_change_password: bool
        :param allow_users_to_change_password: Allows all IAM users in your account
            to use the AWS Management Console to change their own passwords.

        :type hard_expiry: bool
        :param hard_expiry: Prevents IAM users from setting a new password after
            their password has expired.

        :type max_password_age: int
        :param max_password_age: The number of days that an IAM user password is valid.

        :type minimum_password_length: int
        :param minimum_password_length: The minimum number of characters allowed in
            an IAM user password.

        :type password_reuse_prevention: int
        :param password_reuse_prevention: Specifies the number of previous passwords
            that IAM users are prevented from reusing.

        :type require_lowercase_characters: bool
        :param require_lowercase_characters: Specifies whether IAM user passwords
            must contain at least one lowercase character from the ISO basic Latin
            alphabet (``a`` to ``z``).

        :type require_numbers: bool
        :param require_numbers: Specifies whether IAM user passwords must contain at
            least one numeric character (``0`` to ``9``).

        :type require_symbols: bool
        :param require_symbols: Specifies whether IAM user passwords must contain at
            least one of the following non-alphanumeric characters:
            ``! @ # $ % ^ & * ( ) _ + - = [ ] { } | '``

        :type require_uppercase_characters: bool
        :param require_uppercase_characters: Specifies whether IAM user passwords
            must contain at least one uppercase character from the ISO basic Latin
            alphabet (``A`` to ``Z``).
        """
    params = {}
    if allow_users_to_change_password is not None and type(allow_users_to_change_password) is bool:
        params['AllowUsersToChangePassword'] = str(allow_users_to_change_password).lower()
    if hard_expiry is not None and type(allow_users_to_change_password) is bool:
        params['HardExpiry'] = str(hard_expiry).lower()
    if max_password_age is not None:
        params['MaxPasswordAge'] = max_password_age
    if minimum_password_length is not None:
        params['MinimumPasswordLength'] = minimum_password_length
    if password_reuse_prevention is not None:
        params['PasswordReusePrevention'] = password_reuse_prevention
    if require_lowercase_characters is not None and type(allow_users_to_change_password) is bool:
        params['RequireLowercaseCharacters'] = str(require_lowercase_characters).lower()
    if require_numbers is not None and type(allow_users_to_change_password) is bool:
        params['RequireNumbers'] = str(require_numbers).lower()
    if require_symbols is not None and type(allow_users_to_change_password) is bool:
        params['RequireSymbols'] = str(require_symbols).lower()
    if require_uppercase_characters is not None and type(allow_users_to_change_password) is bool:
        params['RequireUppercaseCharacters'] = str(require_uppercase_characters).lower()
    return self.get_response('UpdateAccountPasswordPolicy', params)