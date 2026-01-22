import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
class GPGStrategy:
    """GPG Signing and checking facilities."""
    acceptable_keys: Optional[List[str]] = None

    def __init__(self, config_stack):
        self._config_stack = config_stack
        try:
            import gpg
            self.context = gpg.Context()
            self.context.armor = True
            self.context.signers = self._get_signing_keys()
        except ModuleNotFoundError:
            pass

    def _get_signing_keys(self):
        import gpg
        keyname = self._config_stack.get('gpg_signing_key')
        if keyname == 'default':
            return []
        if keyname:
            try:
                return [self.context.get_key(keyname)]
            except gpg.errors.KeyNotFound:
                pass
        if keyname is None:
            keyname = config.extract_email_address(self._config_stack.get('email'))
        if keyname == 'default':
            return []
        possible_keys = self.context.keylist(keyname, secret=True)
        try:
            return [next(possible_keys)]
        except StopIteration:
            return []

    @staticmethod
    def verify_signatures_available():
        """
        check if this strategy can verify signatures

        :return: boolean if this strategy can verify signatures
        """
        try:
            import gpg
            return True
        except ModuleNotFoundError:
            return False

    def sign(self, content, mode):
        try:
            import gpg
        except ModuleNotFoundError as error:
            raise GpgNotInstalled('Set create_signatures=no to disable creating signatures.')
        if isinstance(content, str):
            raise errors.BzrBadParameterUnicode('content')
        plain_text = gpg.Data(content)
        try:
            output, result = self.context.sign(plain_text, mode={MODE_DETACH: gpg.constants.sig.mode.DETACH, MODE_CLEAR: gpg.constants.sig.mode.CLEAR, MODE_NORMAL: gpg.constants.sig.mode.NORMAL}[mode])
        except gpg.errors.GPGMEError as error:
            raise SigningFailed(str(error))
        except gpg.errors.InvalidSigners as error:
            raise SigningFailed(str(error))
        return output

    def verify(self, signed_data, signature=None):
        """Check content has a valid signature.

        :param signed_data; Signed data
        :param signature: optional signature (if detached)

        :return: SIGNATURE_VALID or a failed SIGNATURE_ value, key uid if valid, plain text
        """
        try:
            import gpg
        except ModuleNotFoundError as error:
            raise GpgNotInstalled('Set check_signatures=ignore to disable verifying signatures.')
        signed_data = gpg.Data(signed_data)
        if signature:
            signature = gpg.Data(signature)
        try:
            plain_output, result = self.context.verify(signed_data, signature)
        except gpg.errors.BadSignatures as error:
            fingerprint = error.result.signatures[0].fpr
            if error.result.signatures[0].summary & gpg.constants.SIGSUM_KEY_EXPIRED:
                expires = self.context.get_key(error.result.signatures[0].fpr).subkeys[0].expires
                if expires > error.result.signatures[0].timestamp:
                    return (SIGNATURE_EXPIRED, fingerprint[-8:], None)
                else:
                    return (SIGNATURE_NOT_VALID, None, None)
            if error.result.signatures[0].summary & gpg.constants.SIGSUM_KEY_MISSING:
                return (SIGNATURE_KEY_MISSING, fingerprint[-8:], None)
            return (SIGNATURE_NOT_VALID, None, None)
        except gpg.errors.GPGMEError as error:
            raise SignatureVerificationFailed(error)
        if len(result.signatures) == 0:
            return (SIGNATURE_NOT_VALID, None, plain_output)
        fingerprint = result.signatures[0].fpr
        if self.acceptable_keys is not None:
            if fingerprint not in self.acceptable_keys:
                return (SIGNATURE_KEY_MISSING, fingerprint[-8:], plain_output)
        if result.signatures[0].summary & gpg.constants.SIGSUM_VALID:
            key = self.context.get_key(fingerprint)
            name = key.uids[0].name
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            email = key.uids[0].email
            if isinstance(email, bytes):
                email = email.decode('utf-8')
            return (SIGNATURE_VALID, name + ' <' + email + '>', plain_output)
        if result.signatures[0].summary & gpg.constants.SIGSUM_RED:
            return (SIGNATURE_NOT_VALID, None, plain_output)
        if result.signatures[0].summary == 0 and self.acceptable_keys is not None:
            if fingerprint in self.acceptable_keys:
                return (SIGNATURE_VALID, None, plain_output)
        if result.signatures[0].summary == 0 and self.acceptable_keys is None:
            return (SIGNATURE_NOT_VALID, None, plain_output)
        raise SignatureVerificationFailed('Unknown GnuPG key verification result')

    def set_acceptable_keys(self, command_line_input):
        """Set the acceptable keys for verifying with this GPGStrategy.

        :param command_line_input: comma separated list of patterns from
                                command line
        :return: nothing
        """
        patterns = None
        acceptable_keys_config = self._config_stack.get('acceptable_keys')
        if acceptable_keys_config is not None:
            patterns = acceptable_keys_config
        if command_line_input is not None:
            patterns = command_line_input.split(',')
        if patterns:
            self.acceptable_keys = []
            for pattern in patterns:
                result = self.context.keylist(pattern)
                found_key = False
                for key in result:
                    found_key = True
                    self.acceptable_keys.append(key.subkeys[0].fpr)
                    trace.mutter('Added acceptable key: ' + key.subkeys[0].fpr)
                if not found_key:
                    trace.note(gettext('No GnuPG key results for pattern: {0}').format(pattern))