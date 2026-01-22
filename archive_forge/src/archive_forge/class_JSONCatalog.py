import json
import os
import re
from pathlib import Path
from django.apps import apps
from django.conf import settings
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.template import Context, Engine
from django.urls import translate_url
from django.utils.formats import get_format
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.translation import check_for_language, get_language
from django.utils.translation.trans_real import DjangoTranslation
from django.views.generic import View
class JSONCatalog(JavaScriptCatalog):
    """
    Return the selected language catalog as a JSON object.

    Receive the same parameters as JavaScriptCatalog and return a response
    with a JSON object of the following format:

        {
            "catalog": {
                # Translations catalog
            },
            "formats": {
                # Language formats for date, time, etc.
            },
            "plural": '...'  # Expression for plural forms, or null.
        }
    """

    def render_to_response(self, context, **response_kwargs):
        return JsonResponse(context)