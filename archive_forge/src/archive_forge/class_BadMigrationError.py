from django.db import DatabaseError
class BadMigrationError(Exception):
    """There's a bad migration (unreadable/bad format/etc.)."""
    pass