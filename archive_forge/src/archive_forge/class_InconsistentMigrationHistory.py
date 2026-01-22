from django.db import DatabaseError
class InconsistentMigrationHistory(Exception):
    """An applied migration has some of its dependencies not applied."""
    pass