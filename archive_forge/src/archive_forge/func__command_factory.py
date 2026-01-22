import time
from argparse import ArgumentParser
from typing import Optional
from huggingface_hub import HfApi, create_branch, get_repo_discussions
from datasets import get_dataset_config_names, get_dataset_default_config_name, load_dataset
from datasets.commands import BaseDatasetsCLICommand
def _command_factory(args):
    return ConvertToParquetCommand(args.dataset_id, args.token, args.revision, args.trust_remote_code)