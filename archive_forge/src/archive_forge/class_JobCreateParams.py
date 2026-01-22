from __future__ import annotations
from typing import Union, Optional
from typing_extensions import Literal, Required, TypedDict
class JobCreateParams(TypedDict, total=False):
    model: Required[Union[str, Literal['babbage-002', 'davinci-002', 'gpt-3.5-turbo']]]
    'The name of the model to fine-tune.\n\n    You can select one of the\n    [supported models](https://platform.openai.com/docs/guides/fine-tuning/what-models-can-be-fine-tuned).\n    '
    training_file: Required[str]
    'The ID of an uploaded file that contains training data.\n\n    See [upload file](https://platform.openai.com/docs/api-reference/files/upload)\n    for how to upload a file.\n\n    Your dataset must be formatted as a JSONL file. Additionally, you must upload\n    your file with the purpose `fine-tune`.\n\n    See the [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)\n    for more details.\n    '
    hyperparameters: Hyperparameters
    'The hyperparameters used for the fine-tuning job.'
    suffix: Optional[str]
    '\n    A string of up to 18 characters that will be added to your fine-tuned model\n    name.\n\n    For example, a `suffix` of "custom-model-name" would produce a model name like\n    `ft:gpt-3.5-turbo:openai:custom-model-name:7p4lURel`.\n    '
    validation_file: Optional[str]
    'The ID of an uploaded file that contains validation data.\n\n    If you provide this file, the data is used to generate validation metrics\n    periodically during fine-tuning. These metrics can be viewed in the fine-tuning\n    results file. The same data should not be present in both train and validation\n    files.\n\n    Your dataset must be formatted as a JSONL file. You must upload your file with\n    the purpose `fine-tune`.\n\n    See the [fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)\n    for more details.\n    '