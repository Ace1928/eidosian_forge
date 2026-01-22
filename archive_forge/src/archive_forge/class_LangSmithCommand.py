import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union, cast
from langsmith import env as ls_env
from langsmith import utils as ls_utils
class LangSmithCommand:
    """Manage the LangSmith Tracing server."""

    def __init__(self) -> None:
        self.docker_compose_file = Path(__file__).absolute().parent / 'docker-compose.yaml'

    @property
    def docker_compose_command(self) -> List[str]:
        return ls_utils.get_docker_compose_command()

    def _open_browser(self, url: str) -> None:
        try:
            subprocess.run(['open', url])
        except FileNotFoundError:
            pass

    def _start_local(self) -> None:
        command = [*self.docker_compose_command, '-f', str(self.docker_compose_file)]
        subprocess.run([*command, 'up', '--quiet-pull', '--wait'])
        logger.info('LangSmith server is running at http://localhost:80/api.\nTo view the app, navigate your browser to http://localhost:80\n\nTo connect your LangChain application to the server locally,\nset the following environment variable when running your LangChain application.\n')
        logger.info('\tLANGSMITH_TRACING=true')
        logger.info('\tLANGSMITH_ENDPOINT=http://localhost:80/api\n')
        self._open_browser('http://localhost')

    def pull(self, *, version: str='0.2.17') -> None:
        """Pull the latest LangSmith images.

        Args:
            version: The LangSmith version to use for LangSmith. Defaults to 0.2.17
        """
        os.environ['_LANGSMITH_IMAGE_VERSION'] = version
        subprocess.run([*self.docker_compose_command, '-f', str(self.docker_compose_file), 'pull'])

    def start(self, *, openai_api_key: Optional[str]=None, langsmith_license_key: str, version: str='0.2.17') -> None:
        """Run the LangSmith server locally.

        Args:
            openai_api_key: The OpenAI API key to use for LangSmith
                If not provided, the OpenAI API Key will be read from the
                OPENAI_API_KEY environment variable. If neither are provided,
                some features of LangSmith will not be available.
            langsmith_license_key: The LangSmith license key to use for LangSmith
                If not provided, the LangSmith license key will be read from the
                LANGSMITH_LICENSE_KEY environment variable. If neither are provided,
                Langsmith will not start up.
            version: The LangSmith version to use for LangSmith. Defaults to latest.
        """
        if openai_api_key is not None:
            os.environ['OPENAI_API_KEY'] = openai_api_key
        if langsmith_license_key is not None:
            os.environ['LANGSMITH_LICENSE_KEY'] = langsmith_license_key
        self.pull(version=version)
        self._start_local()

    def stop(self, clear_volumes: bool=False) -> None:
        """Stop the LangSmith server."""
        cmd = [*self.docker_compose_command, '-f', str(self.docker_compose_file), 'down']
        if clear_volumes:
            confirm = input('You are about to delete all the locally cached LangSmith containers and volumes. This operation cannot be undone. Are you sure? [y/N]')
            if confirm.lower() != 'y':
                print('Aborting.')
                return
            cmd.append('--volumes')
        subprocess.run(cmd)

    def logs(self) -> None:
        """Print the logs from the LangSmith server."""
        subprocess.run([*self.docker_compose_command, '-f', str(self.docker_compose_file), 'logs'])

    def status(self) -> None:
        """Provide information about the status LangSmith server."""
        command = [*self.docker_compose_command, '-f', str(self.docker_compose_file), 'ps', '--format', 'json']
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            command_stdout = result.stdout.decode('utf-8')
            services_status = json.loads(command_stdout)
        except json.JSONDecodeError:
            logger.error('Error checking LangSmith server status.')
            return
        if services_status:
            logger.info('The LangSmith server is currently running.')
            pprint_services(services_status)
        else:
            logger.info('The LangSmith server is not running.')
            return