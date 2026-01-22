import os
import requests
from keystoneauth1 import loading
from keystoneauth1 import plugin
from oslo_log import log
class VitrageKeycloakLoader(loading.BaseLoader):
    plugin_class = VitrageKeycloakPlugin

    def get_options(self):
        options = super(VitrageKeycloakLoader, self).get_options()
        options.extend([VitrageOpt('username', help='User Name', required=True), VitrageOpt('password', help='password', required=True), VitrageOpt('realm-name', help='Realm Name', required=True), VitrageOpt('endpoint', help='Vitrage Endpoint', required=True), VitrageOpt('auth-url', help='Keycloak Url', required=True), VitrageOpt('openid-client-id', help='Keycloak client id', required=True)])
        return options