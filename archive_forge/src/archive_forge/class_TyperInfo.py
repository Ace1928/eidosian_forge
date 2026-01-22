import inspect
import io
from typing import (
import click
import click.shell_completion
class TyperInfo:

    def __init__(self, typer_instance: Optional['Typer']=Default(None), *, name: Optional[str]=Default(None), cls: Optional[Type['TyperGroup']]=Default(None), invoke_without_command: bool=Default(False), no_args_is_help: bool=Default(False), subcommand_metavar: Optional[str]=Default(None), chain: bool=Default(False), result_callback: Optional[Callable[..., Any]]=Default(None), context_settings: Optional[Dict[Any, Any]]=Default(None), callback: Optional[Callable[..., Any]]=Default(None), help: Optional[str]=Default(None), epilog: Optional[str]=Default(None), short_help: Optional[str]=Default(None), options_metavar: str=Default('[OPTIONS]'), add_help_option: bool=Default(True), hidden: bool=Default(False), deprecated: bool=Default(False), rich_help_panel: Union[str, None]=Default(None)):
        self.typer_instance = typer_instance
        self.name = name
        self.cls = cls
        self.invoke_without_command = invoke_without_command
        self.no_args_is_help = no_args_is_help
        self.subcommand_metavar = subcommand_metavar
        self.chain = chain
        self.result_callback = result_callback
        self.context_settings = context_settings
        self.callback = callback
        self.help = help
        self.epilog = epilog
        self.short_help = short_help
        self.options_metavar = options_metavar
        self.add_help_option = add_help_option
        self.hidden = hidden
        self.deprecated = deprecated
        self.rich_help_panel = rich_help_panel