import os
from subprocess import PIPE, STDOUT, Popen
from typing import Optional, Union
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DOCKER_OPENJDK_VERSION
from mlflow.utils import env_manager as em
from mlflow.utils.file_utils import _copy_project
from mlflow.utils.logging_utils import eprint
from mlflow.version import VERSION
def _get_maven_proxy():
    http_proxy = os.getenv('http_proxy')
    https_proxy = os.getenv('https_proxy')
    if not http_proxy or not https_proxy:
        return ''
    parsed_http_proxy = urlparse(http_proxy)
    assert parsed_http_proxy.hostname is not None, 'Invalid `http_proxy` hostname.'
    assert parsed_http_proxy.port is not None, f'Invalid proxy port: {parsed_http_proxy.port}'
    parsed_https_proxy = urlparse(https_proxy)
    assert parsed_https_proxy.hostname is not None, 'Invalid `https_proxy` hostname.'
    assert parsed_https_proxy.port is not None, f'Invalid proxy port: {parsed_https_proxy.port}'
    maven_proxy_options = ('-DproxySet=true', f'-Dhttp.proxyHost={parsed_http_proxy.hostname}', f'-Dhttp.proxyPort={parsed_http_proxy.port}', f'-Dhttps.proxyHost={parsed_https_proxy.hostname}', f'-Dhttps.proxyPort={parsed_https_proxy.port}', '-Dhttps.nonProxyHosts=repo.maven.apache.org')
    if parsed_http_proxy.username is None or parsed_http_proxy.password is None:
        return ' '.join(maven_proxy_options)
    return ' '.join((*maven_proxy_options, f'-Dhttp.proxyUser={parsed_http_proxy.username}', f'-Dhttp.proxyPassword={parsed_http_proxy.password}'))