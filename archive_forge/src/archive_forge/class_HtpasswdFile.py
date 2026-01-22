from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import ExpectedStringError
from passlib.hash import htdigest
from passlib.utils import render_bytes, to_bytes, is_ascii_codec
from passlib.utils.decor import deprecated_method
from passlib.utils.compat import join_bytes, unicode, BytesIO, PY3
class HtpasswdFile(_CommonFile):
    """class for reading & writing Htpasswd files.

    The class constructor accepts the following arguments:

    :type path: filepath
    :param path:

        Specifies path to htpasswd file, use to implicitly load from and save to.

        This class has two modes of operation:

        1. It can be "bound" to a local file by passing a ``path`` to the class
           constructor. In this case it will load the contents of the file when
           created, and the :meth:`load` and :meth:`save` methods will automatically
           load from and save to that file if they are called without arguments.

        2. Alternately, it can exist as an independant object, in which case
           :meth:`load` and :meth:`save` will require an explicit path to be
           provided whenever they are called. As well, ``autosave`` behavior
           will not be available.

           This feature is new in Passlib 1.6, and is the default if no
           ``path`` value is provided to the constructor.

        This is also exposed as a readonly instance attribute.

    :type new: bool
    :param new:

        Normally, if *path* is specified, :class:`HtpasswdFile` will
        immediately load the contents of the file. However, when creating
        a new htpasswd file, applications can set ``new=True`` so that
        the existing file (if any) will not be loaded.

        .. versionadded:: 1.6
            This feature was previously enabled by setting ``autoload=False``.
            That alias has been deprecated, and will be removed in Passlib 1.8

    :type autosave: bool
    :param autosave:

        Normally, any changes made to an :class:`HtpasswdFile` instance
        will not be saved until :meth:`save` is explicitly called. However,
        if ``autosave=True`` is specified, any changes made will be
        saved to disk immediately (assuming *path* has been set).

        This is also exposed as a writeable instance attribute.

    :type encoding: str
    :param encoding:

        Optionally specify character encoding used to read/write file
        and hash passwords. Defaults to ``utf-8``, though ``latin-1``
        is the only other commonly encountered encoding.

        This is also exposed as a readonly instance attribute.

    :type default_scheme: str
    :param default_scheme:
        Optionally specify default scheme to use when encoding new passwords.

        This can be any of the schemes with builtin Apache support,
        OR natively supported by the host OS's :func:`crypt.crypt` function.

        * Builtin schemes include ``"bcrypt"`` (apache 2.4+), ``"apr_md5_crypt"`,
          and ``"des_crypt"``.

        * Schemes commonly supported by Unix hosts
          include ``"bcrypt"``, ``"sha256_crypt"``, and ``"des_crypt"``.

        In order to not have to sort out what you should use,
        passlib offers a number of aliases, that will resolve
        to the most appropriate scheme based on your needs:

        * ``"portable"``, ``"portable_apache_24"`` -- pick scheme that's portable across hosts
          running apache >= 2.4. **This will be the default as of Passlib 2.0**.

        * ``"portable_apache_22"`` -- pick scheme that's portable across hosts
          running apache >= 2.4. **This is the default up to Passlib 1.9**.

        * ``"host"``, ``"host_apache_24"`` -- pick strongest scheme supported by
           apache >= 2.4 and/or host OS.

        * ``"host_apache_22"`` -- pick strongest scheme supported by
           apache >= 2.2 and/or host OS.

        .. versionadded:: 1.6
            This keyword was previously named ``default``. That alias
            has been deprecated, and will be removed in Passlib 1.8.

        .. versionchanged:: 1.6.3

            Added support for ``"bcrypt"``, ``"sha256_crypt"``, and ``"portable"`` alias.

        .. versionchanged:: 1.7

            Added apache 2.4 semantics, and additional aliases.

    :type context: :class:`~passlib.context.CryptContext`
    :param context:
        :class:`!CryptContext` instance used to create
        and verify the hashes found in the htpasswd file.
        The default value is a pre-built context which supports all
        of the hashes officially allowed in an htpasswd file.

        This is also exposed as a readonly instance attribute.

        .. warning::

            This option may be used to add support for non-standard hash
            formats to an htpasswd file. However, the resulting file
            will probably not be usable by another application,
            and particularly not by Apache.

    :param autoload:
        Set to ``False`` to prevent the constructor from automatically
        loaded the file from disk.

        .. deprecated:: 1.6
            This has been replaced by the *new* keyword.
            Instead of setting ``autoload=False``, you should use
            ``new=True``. Support for this keyword will be removed
            in Passlib 1.8.

    :param default:
        Change the default algorithm used to hash new passwords.

        .. deprecated:: 1.6
            This has been renamed to *default_scheme* for clarity.
            Support for this alias will be removed in Passlib 1.8.

    Loading & Saving
    ================
    .. automethod:: load
    .. automethod:: load_if_changed
    .. automethod:: load_string
    .. automethod:: save
    .. automethod:: to_string

    Inspection
    ================
    .. automethod:: users
    .. automethod:: check_password
    .. automethod:: get_hash

    Modification
    ================
    .. automethod:: set_password
    .. automethod:: delete

    Alternate Constructors
    ======================
    .. automethod:: from_string

    Attributes
    ==========
    .. attribute:: path

        Path to local file that will be used as the default
        for all :meth:`load` and :meth:`save` operations.
        May be written to, initialized by the *path* constructor keyword.

    .. attribute:: autosave

        Writeable flag indicating whether changes will be automatically
        written to *path*.

    Errors
    ======
    :raises ValueError:
        All of the methods in this class will raise a :exc:`ValueError` if
        any user name contains a forbidden character (one of ``:\\r\\n\\t\\x00``),
        or is longer than 255 characters.
    """

    def __init__(self, path=None, default_scheme=None, context=htpasswd_context, **kwds):
        if 'default' in kwds:
            warn('``default`` is deprecated as of Passlib 1.6, and will be removed in Passlib 1.8, it has been renamed to ``default_scheem``.', DeprecationWarning, stacklevel=2)
            default_scheme = kwds.pop('default')
        if default_scheme:
            if default_scheme in _warn_no_bcrypt:
                warn('HtpasswdFile: no bcrypt backends available, using fallback for default scheme %r' % default_scheme, exc.PasslibSecurityWarning)
            default_scheme = htpasswd_defaults.get(default_scheme, default_scheme)
            context = context.copy(default=default_scheme)
        self.context = context
        super(HtpasswdFile, self).__init__(path, **kwds)

    def _parse_record(self, record, lineno):
        result = record.rstrip().split(_BCOLON)
        if len(result) != 2:
            raise ValueError('malformed htpasswd file (error reading line %d)' % lineno)
        return result

    def _render_record(self, user, hash):
        return render_bytes('%s:%s\n', user, hash)

    def users(self):
        """
        Return list of all users in database
        """
        return [self._decode_field(user) for user in self._records]

    def set_password(self, user, password):
        """Set password for user; adds user if needed.

        :returns:
            * ``True`` if existing user was updated.
            * ``False`` if user account was added.

        .. versionchanged:: 1.6
            This method was previously called ``update``, it was renamed
            to prevent ambiguity with the dictionary method.
            The old alias is deprecated, and will be removed in Passlib 1.8.
        """
        hash = self.context.hash(password)
        return self.set_hash(user, hash)

    @deprecated_method(deprecated='1.6', removed='1.8', replacement='set_password')
    def update(self, user, password):
        """set password for user"""
        return self.set_password(user, password)

    def get_hash(self, user):
        """Return hash stored for user, or ``None`` if user not found.

        .. versionchanged:: 1.6
            This method was previously named ``find``, it was renamed
            for clarity. The old name is deprecated, and will be removed
            in Passlib 1.8.
        """
        try:
            return self._records[self._encode_user(user)]
        except KeyError:
            return None

    def set_hash(self, user, hash):
        """
        semi-private helper which allows writing a hash directly;
        adds user if needed.

        .. warning::
            does not (currently) do any validation of the hash string

        .. versionadded:: 1.7
        """
        if PY3 and isinstance(hash, str):
            hash = hash.encode(self.encoding)
        user = self._encode_user(user)
        existing = self._set_record(user, hash)
        self._autosave()
        return existing

    @deprecated_method(deprecated='1.6', removed='1.8', replacement='get_hash')
    def find(self, user):
        """return hash for user"""
        return self.get_hash(user)

    def delete(self, user):
        """Delete user's entry.

        :returns:
            * ``True`` if user deleted.
            * ``False`` if user not found.
        """
        try:
            del self._records[self._encode_user(user)]
        except KeyError:
            return False
        self._autosave()
        return True

    def check_password(self, user, password):
        """
        Verify password for specified user.
        If algorithm marked as deprecated by CryptContext, will automatically be re-hashed.

        :returns:
            * ``None`` if user not found.
            * ``False`` if user found, but password does not match.
            * ``True`` if user found and password matches.

        .. versionchanged:: 1.6
            This method was previously called ``verify``, it was renamed
            to prevent ambiguity with the :class:`!CryptContext` method.
            The old alias is deprecated, and will be removed in Passlib 1.8.
        """
        user = self._encode_user(user)
        hash = self._records.get(user)
        if hash is None:
            return None
        if isinstance(password, unicode):
            password = password.encode(self.encoding)
        ok, new_hash = self.context.verify_and_update(password, hash)
        if ok and new_hash is not None:
            assert user in self._records
            self._records[user] = new_hash
            self._autosave()
        return ok

    @deprecated_method(deprecated='1.6', removed='1.8', replacement='check_password')
    def verify(self, user, password):
        """verify password for user"""
        return self.check_password(user, password)