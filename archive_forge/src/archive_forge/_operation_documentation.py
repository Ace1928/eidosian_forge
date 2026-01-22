from ._auth_context import ContextKey
 Returns a context(AuthContext) which is associated with all the given
    operations (list of string). It will be based on the auth context
    passed in as ctx.

    An allow caveat will succeed only if one of the allowed operations is in
    ops; a deny caveat will succeed only if none of the denied operations are
    in ops.
    