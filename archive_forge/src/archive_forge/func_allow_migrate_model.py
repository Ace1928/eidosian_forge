from django.db import router
def allow_migrate_model(self, connection_alias, model):
    """
        Return whether or not a model may be migrated.

        This is a thin wrapper around router.allow_migrate_model() that
        preemptively rejects any proxy, swapped out, or unmanaged model.
        """
    if not model._meta.can_migrate(connection_alias):
        return False
    return router.allow_migrate_model(connection_alias, model)