from django.conf import settings
from django.contrib.sessions.backends.db import SessionStore as DBStore
from django.core.cache import caches

        Remove the current session data from the database and regenerate the
        key.
        