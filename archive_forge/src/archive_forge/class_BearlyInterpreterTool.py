import base64
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Type
import requests
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import Tool
class BearlyInterpreterTool:
    """Tool for evaluating python code in a sandbox environment."""
    api_key: str
    endpoint = 'https://exec.bearly.ai/v1/interpreter'
    name = 'bearly_interpreter'
    args_schema: Type[BaseModel] = BearlyInterpreterToolArguments
    files: Dict[str, FileInfo] = {}

    def __init__(self, api_key: str):
        self.api_key = api_key

    @property
    def file_description(self) -> str:
        if len(self.files) == 0:
            return ''
        lines = ['The following files available in the evaluation environment:']
        for target_path, file_info in self.files.items():
            peek_content = head_file(file_info.source_path, 4)
            lines.append(f'- path: `{target_path}` \n first four lines: {peek_content} \n description: `{file_info.description}`')
        return '\n'.join(lines)

    @property
    def description(self) -> str:
        return (base_description + '\n\n' + self.file_description).strip()

    def make_input_files(self) -> List[dict]:
        files = []
        for target_path, file_info in self.files.items():
            files.append({'pathname': target_path, 'contentsBasesixtyfour': file_to_base64(file_info.source_path)})
        return files

    def _run(self, python_code: str) -> dict:
        script = strip_markdown_code(python_code)
        resp = requests.post('https://exec.bearly.ai/v1/interpreter', data=json.dumps({'fileContents': script, 'inputFiles': self.make_input_files(), 'outputDir': 'output/', 'outputAsLinks': True}), headers={'Authorization': self.api_key}).json()
        return {'stdout': base64.b64decode(resp['stdoutBasesixtyfour']).decode() if resp['stdoutBasesixtyfour'] else '', 'stderr': base64.b64decode(resp['stderrBasesixtyfour']).decode() if resp['stderrBasesixtyfour'] else '', 'fileLinks': resp['fileLinks'], 'exitCode': resp['exitCode']}

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError('custom_search does not support async')

    def add_file(self, source_path: str, target_path: str, description: str) -> None:
        if target_path in self.files:
            raise ValueError('target_path already exists')
        if not Path(source_path).exists():
            raise ValueError('source_path does not exist')
        self.files[target_path] = FileInfo(target_path=target_path, source_path=source_path, description=description)

    def clear_files(self) -> None:
        self.files = {}

    def as_tool(self) -> Tool:
        return Tool.from_function(func=self._run, name=self.name, description=self.description, args_schema=self.args_schema)