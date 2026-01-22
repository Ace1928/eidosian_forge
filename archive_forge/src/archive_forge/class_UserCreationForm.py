import unicodedata
from django import forms
from django.contrib.auth import authenticate, get_user_model, password_validation
from django.contrib.auth.hashers import UNUSABLE_PASSWORD_PREFIX, identify_hasher
from django.contrib.auth.models import User
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.core.exceptions import ValidationError
from django.core.mail import EmailMultiAlternatives
from django.template import loader
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode
from django.utils.text import capfirst
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
class UserCreationForm(BaseUserCreationForm):

    def clean_username(self):
        """Reject usernames that differ only in case."""
        username = self.cleaned_data.get('username')
        if username and self._meta.model.objects.filter(username__iexact=username).exists():
            self._update_errors(ValidationError({'username': self.instance.unique_error_message(self._meta.model, ['username'])}))
        else:
            return username