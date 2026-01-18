import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
import boto3
import yaml
from google.cloud import storage
import ray

    Run the necessary Ray commands to start a cluster, verify Ray is running, and clean
    up the cluster.

    Args:
        cluster_config: The path of the cluster configuration file.
        retries: The number of retries for the verification step.
        no_config_cache: Whether to pass the --no-config-cache flag to the ray CLI
            commands.
    