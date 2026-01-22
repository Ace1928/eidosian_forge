import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class ACLCommands(CommandsProtocol):
    """
    Redis Access Control List (ACL) commands.
    see: https://redis.io/topics/acl
    """

    def acl_cat(self, category: Union[str, None]=None, **kwargs) -> ResponseT:
        """
        Returns a list of categories or commands within a category.

        If ``category`` is not supplied, returns a list of all categories.
        If ``category`` is supplied, returns a list of all commands within
        that category.

        For more information see https://redis.io/commands/acl-cat
        """
        pieces: list[EncodableT] = [category] if category else []
        return self.execute_command('ACL CAT', *pieces, **kwargs)

    def acl_dryrun(self, username, *args, **kwargs):
        """
        Simulate the execution of a given command by a given ``username``.

        For more information see https://redis.io/commands/acl-dryrun
        """
        return self.execute_command('ACL DRYRUN', username, *args, **kwargs)

    def acl_deluser(self, *username: str, **kwargs) -> ResponseT:
        """
        Delete the ACL for the specified ``username``s

        For more information see https://redis.io/commands/acl-deluser
        """
        return self.execute_command('ACL DELUSER', *username, **kwargs)

    def acl_genpass(self, bits: Union[int, None]=None, **kwargs) -> ResponseT:
        """Generate a random password value.
        If ``bits`` is supplied then use this number of bits, rounded to
        the next multiple of 4.
        See: https://redis.io/commands/acl-genpass
        """
        pieces = []
        if bits is not None:
            try:
                b = int(bits)
                if b < 0 or b > 4096:
                    raise ValueError
                pieces.append(b)
            except ValueError:
                raise DataError('genpass optionally accepts a bits argument, between 0 and 4096.')
        return self.execute_command('ACL GENPASS', *pieces, **kwargs)

    def acl_getuser(self, username: str, **kwargs) -> ResponseT:
        """
        Get the ACL details for the specified ``username``.

        If ``username`` does not exist, return None

        For more information see https://redis.io/commands/acl-getuser
        """
        return self.execute_command('ACL GETUSER', username, **kwargs)

    def acl_help(self, **kwargs) -> ResponseT:
        """The ACL HELP command returns helpful text describing
        the different subcommands.

        For more information see https://redis.io/commands/acl-help
        """
        return self.execute_command('ACL HELP', **kwargs)

    def acl_list(self, **kwargs) -> ResponseT:
        """
        Return a list of all ACLs on the server

        For more information see https://redis.io/commands/acl-list
        """
        return self.execute_command('ACL LIST', **kwargs)

    def acl_log(self, count: Union[int, None]=None, **kwargs) -> ResponseT:
        """
        Get ACL logs as a list.
        :param int count: Get logs[0:count].
        :rtype: List.

        For more information see https://redis.io/commands/acl-log
        """
        args = []
        if count is not None:
            if not isinstance(count, int):
                raise DataError('ACL LOG count must be an integer')
            args.append(count)
        return self.execute_command('ACL LOG', *args, **kwargs)

    def acl_log_reset(self, **kwargs) -> ResponseT:
        """
        Reset ACL logs.
        :rtype: Boolean.

        For more information see https://redis.io/commands/acl-log
        """
        args = [b'RESET']
        return self.execute_command('ACL LOG', *args, **kwargs)

    def acl_load(self, **kwargs) -> ResponseT:
        """
        Load ACL rules from the configured ``aclfile``.

        Note that the server must be configured with the ``aclfile``
        directive to be able to load ACL rules from an aclfile.

        For more information see https://redis.io/commands/acl-load
        """
        return self.execute_command('ACL LOAD', **kwargs)

    def acl_save(self, **kwargs) -> ResponseT:
        """
        Save ACL rules to the configured ``aclfile``.

        Note that the server must be configured with the ``aclfile``
        directive to be able to save ACL rules to an aclfile.

        For more information see https://redis.io/commands/acl-save
        """
        return self.execute_command('ACL SAVE', **kwargs)

    def acl_setuser(self, username: str, enabled: bool=False, nopass: bool=False, passwords: Union[str, Iterable[str], None]=None, hashed_passwords: Union[str, Iterable[str], None]=None, categories: Optional[Iterable[str]]=None, commands: Optional[Iterable[str]]=None, keys: Optional[Iterable[KeyT]]=None, channels: Optional[Iterable[ChannelT]]=None, selectors: Optional[Iterable[Tuple[str, KeyT]]]=None, reset: bool=False, reset_keys: bool=False, reset_channels: bool=False, reset_passwords: bool=False, **kwargs) -> ResponseT:
        """
        Create or update an ACL user.

        Create or update the ACL for ``username``. If the user already exists,
        the existing ACL is completely overwritten and replaced with the
        specified values.

        ``enabled`` is a boolean indicating whether the user should be allowed
        to authenticate or not. Defaults to ``False``.

        ``nopass`` is a boolean indicating whether the can authenticate without
        a password. This cannot be True if ``passwords`` are also specified.

        ``passwords`` if specified is a list of plain text passwords
        to add to or remove from the user. Each password must be prefixed with
        a '+' to add or a '-' to remove. For convenience, the value of
        ``passwords`` can be a simple prefixed string when adding or
        removing a single password.

        ``hashed_passwords`` if specified is a list of SHA-256 hashed passwords
        to add to or remove from the user. Each hashed password must be
        prefixed with a '+' to add or a '-' to remove. For convenience,
        the value of ``hashed_passwords`` can be a simple prefixed string when
        adding or removing a single password.

        ``categories`` if specified is a list of strings representing category
        permissions. Each string must be prefixed with either a '+' to add the
        category permission or a '-' to remove the category permission.

        ``commands`` if specified is a list of strings representing command
        permissions. Each string must be prefixed with either a '+' to add the
        command permission or a '-' to remove the command permission.

        ``keys`` if specified is a list of key patterns to grant the user
        access to. Keys patterns allow '*' to support wildcard matching. For
        example, '*' grants access to all keys while 'cache:*' grants access
        to all keys that are prefixed with 'cache:'. ``keys`` should not be
        prefixed with a '~'.

        ``reset`` is a boolean indicating whether the user should be fully
        reset prior to applying the new ACL. Setting this to True will
        remove all existing passwords, flags and privileges from the user and
        then apply the specified rules. If this is False, the user's existing
        passwords, flags and privileges will be kept and any new specified
        rules will be applied on top.

        ``reset_keys`` is a boolean indicating whether the user's key
        permissions should be reset prior to applying any new key permissions
        specified in ``keys``. If this is False, the user's existing
        key permissions will be kept and any new specified key permissions
        will be applied on top.

        ``reset_channels`` is a boolean indicating whether the user's channel
        permissions should be reset prior to applying any new channel permissions
        specified in ``channels``.If this is False, the user's existing
        channel permissions will be kept and any new specified channel permissions
        will be applied on top.

        ``reset_passwords`` is a boolean indicating whether to remove all
        existing passwords and the 'nopass' flag from the user prior to
        applying any new passwords specified in 'passwords' or
        'hashed_passwords'. If this is False, the user's existing passwords
        and 'nopass' status will be kept and any new specified passwords
        or hashed_passwords will be applied on top.

        For more information see https://redis.io/commands/acl-setuser
        """
        encoder = self.get_encoder()
        pieces: List[EncodableT] = [username]
        if reset:
            pieces.append(b'reset')
        if reset_keys:
            pieces.append(b'resetkeys')
        if reset_channels:
            pieces.append(b'resetchannels')
        if reset_passwords:
            pieces.append(b'resetpass')
        if enabled:
            pieces.append(b'on')
        else:
            pieces.append(b'off')
        if (passwords or hashed_passwords) and nopass:
            raise DataError("Cannot set 'nopass' and supply 'passwords' or 'hashed_passwords'")
        if passwords:
            passwords = list_or_args(passwords, [])
            for i, password in enumerate(passwords):
                password = encoder.encode(password)
                if password.startswith(b'+'):
                    pieces.append(b'>%s' % password[1:])
                elif password.startswith(b'-'):
                    pieces.append(b'<%s' % password[1:])
                else:
                    raise DataError(f'Password {i} must be prefixed with a "+" to add or a "-" to remove')
        if hashed_passwords:
            hashed_passwords = list_or_args(hashed_passwords, [])
            for i, hashed_password in enumerate(hashed_passwords):
                hashed_password = encoder.encode(hashed_password)
                if hashed_password.startswith(b'+'):
                    pieces.append(b'#%s' % hashed_password[1:])
                elif hashed_password.startswith(b'-'):
                    pieces.append(b'!%s' % hashed_password[1:])
                else:
                    raise DataError(f'Hashed password {i} must be prefixed with a "+" to add or a "-" to remove')
        if nopass:
            pieces.append(b'nopass')
        if categories:
            for category in categories:
                category = encoder.encode(category)
                if category.startswith(b'+@'):
                    pieces.append(category)
                elif category.startswith(b'+'):
                    pieces.append(b'+@%s' % category[1:])
                elif category.startswith(b'-@'):
                    pieces.append(category)
                elif category.startswith(b'-'):
                    pieces.append(b'-@%s' % category[1:])
                else:
                    raise DataError(f'Category "{encoder.decode(category, force=True)}" must be prefixed with "+" or "-"')
        if commands:
            for cmd in commands:
                cmd = encoder.encode(cmd)
                if not cmd.startswith(b'+') and (not cmd.startswith(b'-')):
                    raise DataError(f'Command "{encoder.decode(cmd, force=True)}" must be prefixed with "+" or "-"')
                pieces.append(cmd)
        if keys:
            for key in keys:
                key = encoder.encode(key)
                if not key.startswith(b'%') and (not key.startswith(b'~')):
                    key = b'~%s' % key
                pieces.append(key)
        if channels:
            for channel in channels:
                channel = encoder.encode(channel)
                pieces.append(b'&%s' % channel)
        if selectors:
            for cmd, key in selectors:
                cmd = encoder.encode(cmd)
                if not cmd.startswith(b'+') and (not cmd.startswith(b'-')):
                    raise DataError(f'Command "{encoder.decode(cmd, force=True)}" must be prefixed with "+" or "-"')
                key = encoder.encode(key)
                if not key.startswith(b'%') and (not key.startswith(b'~')):
                    key = b'~%s' % key
                pieces.append(b'(%s %s)' % (cmd, key))
        return self.execute_command('ACL SETUSER', *pieces, **kwargs)

    def acl_users(self, **kwargs) -> ResponseT:
        """Returns a list of all registered users on the server.

        For more information see https://redis.io/commands/acl-users
        """
        return self.execute_command('ACL USERS', **kwargs)

    def acl_whoami(self, **kwargs) -> ResponseT:
        """Get the username for the current connection

        For more information see https://redis.io/commands/acl-whoami
        """
        return self.execute_command('ACL WHOAMI', **kwargs)