import logging
from sentry_sdk.integrations.aws_lambda import get_lambda_bootstrap  # type: ignore
Check if we are running in a lambda environment.