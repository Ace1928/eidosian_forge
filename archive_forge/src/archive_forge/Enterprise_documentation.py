import urllib.parse
from typing import Any, Dict
from github.EnterpriseConsumedLicenses import EnterpriseConsumedLicenses
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.Requester import Requester

        :calls: `GET /enterprises/{enterprise}/consumed-licenses <https://docs.github.com/en/enterprise-cloud@latest/rest/enterprise-admin/license#list-enterprise-consumed-licenses>`_
        