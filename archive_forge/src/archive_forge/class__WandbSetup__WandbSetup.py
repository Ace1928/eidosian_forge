import logging
import os
import sys
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import wandb
from . import wandb_manager, wandb_settings
from .lib import config_util, server, tracelog
class _WandbSetup__WandbSetup:
    """Inner class of _WandbSetup."""
    _manager: Optional[wandb_manager._Manager]
    _pid: int

    def __init__(self, pid: int, settings: Optional[Settings]=None, environ: Optional[Dict[str, Any]]=None) -> None:
        self._environ = environ or dict(os.environ)
        self._sweep_config: Optional[Dict[str, Any]] = None
        self._config: Optional[Dict[str, Any]] = None
        self._server: Optional[server.Server] = None
        self._manager: Optional[wandb_manager._Manager] = None
        self._pid = pid
        self._global_run_stack: List[wandb_run.Run] = []
        self._early_logger = _EarlyLogger()
        _set_logger(self._early_logger)
        self._settings = self._settings_setup(settings, self._early_logger)
        wandb.termsetup(self._settings, logger)
        self._check()
        self._setup()
        tracelog_mode = self._settings._tracelog
        if tracelog_mode:
            tracelog.enable(tracelog_mode)

    def _settings_setup(self, settings: Optional[Settings]=None, early_logger: Optional[_EarlyLogger]=None) -> 'wandb_settings.Settings':
        s = wandb_settings.Settings()
        s._apply_base(pid=self._pid, _logger=early_logger)
        s._apply_config_files(_logger=early_logger)
        s._apply_env_vars(self._environ, _logger=early_logger)
        if isinstance(settings, wandb_settings.Settings):
            s._apply_settings(settings, _logger=early_logger)
        elif isinstance(settings, dict):
            s._apply_setup(settings, _logger=early_logger)
        s._infer_settings_from_environment()
        if not s._cli_only_mode:
            s._infer_run_settings_from_environment(_logger=early_logger)
        return s

    def _update(self, settings: Optional[Settings]=None) -> None:
        if settings is None:
            return
        if isinstance(settings, wandb_settings.Settings):
            self._settings._apply_settings(settings)
        elif isinstance(settings, dict):
            self._settings.update(settings, source=wandb_settings.Source.SETUP)

    def _update_user_settings(self, settings: Optional[Settings]=None) -> None:
        settings = settings or self._settings
        self._server = None
        user_settings = self._load_user_settings(settings=settings)
        if user_settings is not None:
            self._settings._apply_user(user_settings)

    def _early_logger_flush(self, new_logger: Logger) -> None:
        if not self._early_logger:
            return
        _set_logger(new_logger)
        self._early_logger._flush()

    def _get_logger(self) -> Optional[Logger]:
        return logger

    @property
    def settings(self) -> 'wandb_settings.Settings':
        return self._settings

    def _get_entity(self) -> Optional[str]:
        if self._settings and self._settings._offline:
            return None
        if self._server is None:
            self._load_viewer()
        assert self._server is not None
        entity = self._server._viewer.get('entity')
        return entity

    def _get_username(self) -> Optional[str]:
        if self._settings and self._settings._offline:
            return None
        if self._server is None:
            self._load_viewer()
        assert self._server is not None
        username = self._server._viewer.get('username')
        return username

    def _get_teams(self) -> List[str]:
        if self._settings and self._settings._offline:
            return []
        if self._server is None:
            self._load_viewer()
        assert self._server is not None
        teams = self._server._viewer.get('teams')
        if teams:
            teams = [team['node']['name'] for team in teams['edges']]
        return teams or []

    def _load_viewer(self, settings: Optional[Settings]=None) -> None:
        if self._settings and self._settings._offline:
            return
        if isinstance(settings, dict):
            settings = wandb_settings.Settings(**settings)
        s = server.Server(settings=settings)
        s.query_with_timeout()
        self._server = s

    def _load_user_settings(self, settings: Optional[Settings]=None) -> Optional[Dict[str, Any]]:
        if self._server is None:
            self._load_viewer(settings=settings)
        if self._server is None:
            return None
        flags = self._server._flags
        user_settings = dict()
        if 'code_saving_enabled' in flags:
            user_settings['save_code'] = flags['code_saving_enabled']
        email = self._server._viewer.get('email', None)
        if email:
            user_settings['email'] = email
        return user_settings

    def _check(self) -> None:
        if hasattr(threading, 'main_thread'):
            if threading.current_thread() is not threading.main_thread():
                pass
        elif threading.current_thread().name != 'MainThread':
            print('bad thread2', threading.current_thread().name)
        if getattr(sys, 'frozen', False):
            print('frozen, could be trouble')

    def _setup(self) -> None:
        self._setup_manager()
        sweep_path = self._settings.sweep_param_path
        if sweep_path:
            self._sweep_config = config_util.dict_from_config_file(sweep_path, must_exist=True)
        if self._settings.config_paths:
            for config_path in self._settings.config_paths:
                config_dict = config_util.dict_from_config_file(config_path)
                if config_dict is None:
                    continue
                if self._config is not None:
                    self._config.update(config_dict)
                else:
                    self._config = config_dict

    def _teardown(self, exit_code: Optional[int]=None) -> None:
        exit_code = exit_code or 0
        self._teardown_manager(exit_code=exit_code)

    def _setup_manager(self) -> None:
        if self._settings._disable_service:
            return
        self._manager = wandb_manager._Manager(settings=self._settings)

    def _teardown_manager(self, exit_code: int) -> None:
        if not self._manager:
            return
        self._manager._teardown(exit_code)
        self._manager = None

    def _get_manager(self) -> Optional[wandb_manager._Manager]:
        return self._manager