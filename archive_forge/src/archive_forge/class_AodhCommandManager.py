import logging
import os
import sys
import warnings
from cliff import app
from cliff import commandmanager
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from aodhclient import __version__
from aodhclient import client
from aodhclient import noauth
from aodhclient.v2 import alarm_cli
from aodhclient.v2 import alarm_history_cli
from aodhclient.v2 import capabilities_cli
class AodhCommandManager(commandmanager.CommandManager):
    SHELL_COMMANDS = {'alarm create': alarm_cli.CliAlarmCreate, 'alarm delete': alarm_cli.CliAlarmDelete, 'alarm list': alarm_cli.CliAlarmList, 'alarm show': alarm_cli.CliAlarmShow, 'alarm update': alarm_cli.CliAlarmUpdate, 'alarm state get': alarm_cli.CliAlarmStateGet, 'alarm state set': alarm_cli.CliAlarmStateSet, 'alarm-history show': alarm_history_cli.CliAlarmHistoryShow, 'alarm-history search': alarm_history_cli.CliAlarmHistorySearch, 'capabilities list': capabilities_cli.CliCapabilitiesList}

    def load_commands(self, namespace):
        for name, command_class in self.SHELL_COMMANDS.items():
            self.add_command(name, command_class)