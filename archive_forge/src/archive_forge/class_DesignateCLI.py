import logging
import os
from tempest.lib.cli import base
from designateclient.functionaltests.config import cfg
from designateclient.functionaltests.models import FieldValueModel
from designateclient.functionaltests.models import ListModel
class DesignateCLI(base.CLIClient, ZoneCommands, ZoneTransferCommands, ZoneExportCommands, ZoneImportCommands, RecordsetCommands, TLDCommands, BlacklistCommands, SharedZoneCommands):
    _CLIENTS = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        resp = FieldValueModel(self.openstack('token issue'))
        self.project_id = resp.project_id

    @property
    def using_auth_override(self):
        return bool(cfg.CONF.identity.override_endpoint)

    @classmethod
    def get_clients(cls):
        if not cls._CLIENTS:
            cls._init_clients()
        return cls._CLIENTS

    @classmethod
    def _init_clients(cls):
        cls._CLIENTS = {'default': DesignateCLI(cli_dir=cfg.CONF.designateclient.directory, username=cfg.CONF.identity.username, password=cfg.CONF.identity.password, tenant_name=cfg.CONF.identity.tenant_name, uri=cfg.CONF.identity.uri), 'alt': DesignateCLI(cli_dir=cfg.CONF.designateclient.directory, username=cfg.CONF.identity.alt_username, password=cfg.CONF.identity.alt_password, tenant_name=cfg.CONF.identity.alt_tenant_name, uri=cfg.CONF.identity.uri), 'admin': DesignateCLI(cli_dir=cfg.CONF.designateclient.directory, username=cfg.CONF.identity.admin_username, password=cfg.CONF.identity.admin_password, tenant_name=cfg.CONF.identity.admin_tenant_name, uri=cfg.CONF.identity.uri)}

    @classmethod
    def as_user(self, user):
        clients = self.get_clients()
        if user in clients:
            return clients[user]
        raise Exception(f"User '{user}' does not exist")

    def parsed_cmd(self, cmd, model=None, *args, **kwargs):
        if self.using_auth_override:
            func = self._openstack_noauth
        else:
            func = self.openstack
        out = func(cmd, *args, **kwargs)
        LOG.debug(out)
        if model is not None:
            return model(out)
        return out

    def _openstack_noauth(self, cmd, *args, **kwargs):
        exe = os.path.join(cfg.CONF.designateclient.directory, 'openstack')
        options = build_option_string({'--os-url': cfg.CONF.identity.override_endpoint, '--os-token': cfg.CONF.identity.override_token})
        cmd = options + ' ' + cmd
        return base.execute(exe, cmd, *args, **kwargs)