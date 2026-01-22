import asyncio
import logging
from typing import Any, Dict, List, Optional, cast
import wandb
from wandb.apis.internal import Api
from wandb.sdk.launch.environment.aws_environment import AwsEnvironment
from wandb.sdk.launch.errors import LaunchError
from .._project_spec import EntryPoint, LaunchProject, get_entry_point_command
from ..builder.build import get_env_vars_dict
from ..registry.abstract import AbstractRegistry
from ..utils import (
from .abstract import AbstractRun, AbstractRunner, Status
class SageMakerRunner(AbstractRunner):
    """Runner class, uses a project to create a SagemakerSubmittedRun."""

    def __init__(self, api: Api, backend_config: Dict[str, Any], environment: AwsEnvironment, registry: AbstractRegistry) -> None:
        """Initialize the SagemakerRunner.

        Arguments:
            api (Api): The API instance.
            backend_config (Dict[str, Any]): The backend configuration.
            environment (AwsEnvironment): The AWS environment.

        Raises:
            LaunchError: If the runner cannot be initialized.
        """
        super().__init__(api, backend_config)
        self.environment = environment
        self.registry = registry

    async def run(self, launch_project: LaunchProject, image_uri: str) -> Optional[AbstractRun]:
        """Run a project on Amazon Sagemaker.

        Arguments:
            launch_project (LaunchProject): The project to run.

        Returns:
            Optional[AbstractRun]: The run instance.

        Raises:
            LaunchError: If the launch is unsuccessful.
        """
        _logger.info('using AWSSagemakerRunner')
        given_sagemaker_args = launch_project.resource_args.get('sagemaker')
        if given_sagemaker_args is None:
            raise LaunchError('No sagemaker args specified. Specify sagemaker args in resource_args')
        default_output_path = self.backend_config.get('runner', {}).get('s3_output_path')
        if default_output_path is not None and (not default_output_path.startswith('s3://')):
            default_output_path = f's3://{default_output_path}'
        session = await self.environment.get_session()
        client = await event_loop_thread_exec(session.client)('sts')
        caller_id = client.get_caller_identity()
        account_id = caller_id['Account']
        _logger.info(f'Using account ID {account_id}')
        role_arn = get_role_arn(given_sagemaker_args, self.backend_config, account_id)
        sagemaker_client = session.client('sagemaker')
        log_client = None
        try:
            log_client = session.client('logs')
        except Exception as e:
            wandb.termwarn(f'Failed to connect to cloudwatch logs with error {str(e)}, logs will not be available')
        if given_sagemaker_args.get('AlgorithmSpecification', {}).get('TrainingImage') is not None:
            sagemaker_args = build_sagemaker_args(launch_project, self._api, role_arn, launch_project.override_entrypoint, launch_project.override_args, MAX_ENV_LENGTHS[self.__class__.__name__], given_sagemaker_args.get('AlgorithmSpecification', {}).get('TrainingImage'), default_output_path)
            _logger.info(f'Launching sagemaker job on user supplied image with args: {sagemaker_args}')
            run = await launch_sagemaker_job(launch_project, sagemaker_args, sagemaker_client, log_client)
            if self.backend_config[PROJECT_SYNCHRONOUS]:
                await run.wait()
            return run
        launch_project.fill_macros(image_uri)
        _logger.info('Connecting to sagemaker client')
        entry_point = launch_project.override_entrypoint or launch_project.get_single_entry_point()
        command_args = get_entry_point_command(entry_point, launch_project.override_args)
        if command_args:
            command_str = ' '.join(command_args)
            wandb.termlog(f'{LOG_PREFIX}Launching run on sagemaker with entrypoint: {command_str}')
        else:
            wandb.termlog(f'{LOG_PREFIX}Launching run on sagemaker with user-provided entrypoint in image')
        sagemaker_args = build_sagemaker_args(launch_project, self._api, role_arn, entry_point, launch_project.override_args, MAX_ENV_LENGTHS[self.__class__.__name__], image_uri, default_output_path)
        _logger.info(f'Launching sagemaker job with args: {sagemaker_args}')
        run = await launch_sagemaker_job(launch_project, sagemaker_args, sagemaker_client, log_client)
        if self.backend_config[PROJECT_SYNCHRONOUS]:
            await run.wait()
        return run