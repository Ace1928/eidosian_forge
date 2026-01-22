from __future__ import annotations
from typing import TYPE_CHECKING
from argparse import ArgumentParser
from .._utils import get_client, print_model
from .._models import BaseModel
class CLIModels:

    @staticmethod
    def get(args: CLIModelIDArgs) -> None:
        model = get_client().models.retrieve(model=args.id)
        print_model(model)

    @staticmethod
    def delete(args: CLIModelIDArgs) -> None:
        model = get_client().models.delete(model=args.id)
        print_model(model)

    @staticmethod
    def list() -> None:
        models = get_client().models.list()
        for model in models:
            print_model(model)