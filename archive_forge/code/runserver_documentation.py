from django.conf import settings
from django.contrib.staticfiles.handlers import StaticFilesHandler
from django.core.management.commands.runserver import Command as RunserverCommand

        Return the static files serving handler wrapping the default handler,
        if static files should be served. Otherwise return the default handler.
        