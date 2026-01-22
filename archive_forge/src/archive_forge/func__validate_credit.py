from __future__ import annotations
from typing import Any, Union
from typing_extensions import TypedDict
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@staticmethod
def _validate_credit(credit: Credit) -> None:
    assert isinstance(credit, (dict, AdvisoryCredit)), credit
    if isinstance(credit, dict):
        assert 'login' in credit, credit
        assert 'type' in credit, credit
        assert isinstance(credit['login'], (str, github.NamedUser.NamedUser)), credit['login']
        assert isinstance(credit['type'], str), credit['type']
    else:
        assert isinstance(credit.login, str), credit.login
        assert isinstance(credit.type, str), credit.type