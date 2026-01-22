import time
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_community.callbacks.utils import import_pandas
class FiddlerCallbackHandler(BaseCallbackHandler):

    def __init__(self, url: str, org: str, project: str, model: str, api_key: str) -> None:
        """
        Initialize Fiddler callback handler.

        Args:
            url: Fiddler URL (e.g. https://demo.fiddler.ai).
                Make sure to include the protocol (http/https).
            org: Fiddler organization id
            project: Fiddler project name to publish events to
            model: Fiddler model name to publish events to
            api_key: Fiddler authentication token
        """
        super().__init__()
        self.fdl = import_fiddler()
        self.pd = import_pandas()
        self.url = url
        self.org = org
        self.project = project
        self.model = model
        self.api_key = api_key
        self._df = self.pd.DataFrame(_dataset_dict)
        self.run_id_prompts: Dict[UUID, List[str]] = {}
        self.run_id_response: Dict[UUID, List[str]] = {}
        self.run_id_starttime: Dict[UUID, int] = {}
        self.fiddler_client = self.fdl.FiddlerApi(url, org_id=org, auth_token=api_key)
        if self.project not in self.fiddler_client.get_project_names():
            print(f'adding project {self.project}.This only has to be done once.')
            try:
                self.fiddler_client.add_project(self.project)
            except Exception as e:
                print(f'Error adding project {self.project}:{{e}}. Fiddler integration will not work.')
                raise e
        dataset_info = self.fdl.DatasetInfo.from_dataframe(self._df, max_inferred_cardinality=0)
        for i in range(len(dataset_info.columns)):
            if dataset_info.columns[i].name == FEEDBACK:
                dataset_info.columns[i].data_type = self.fdl.DataType.CATEGORY
                dataset_info.columns[i].possible_values = FEEDBACK_POSSIBLE_VALUES
            elif dataset_info.columns[i].name == LLM_STATUS:
                dataset_info.columns[i].data_type = self.fdl.DataType.CATEGORY
                dataset_info.columns[i].possible_values = [SUCCESS, FAILURE]
        if self.model not in self.fiddler_client.get_model_names(self.project):
            if self.model not in self.fiddler_client.get_dataset_names(self.project):
                print(f'adding dataset {self.model} to project {self.project}.This only has to be done once.')
                try:
                    self.fiddler_client.upload_dataset(project_id=self.project, dataset_id=self.model, dataset={'train': self._df}, info=dataset_info)
                except Exception as e:
                    print(f'Error adding dataset {self.model}: {e}.Fiddler integration will not work.')
                    raise e
            model_info = self.fdl.ModelInfo.from_dataset_info(dataset_info=dataset_info, dataset_id='train', model_task=self.fdl.ModelTask.LLM, features=[PROMPT, CONTEXT, RESPONSE], target=FEEDBACK, metadata_cols=[RUN_ID, TOTAL_TOKENS, PROMPT_TOKENS, COMPLETION_TOKENS, MODEL_NAME, DURATION], custom_features=self.custom_features)
            print(f'adding model {self.model} to project {self.project}.This only has to be done once.')
            try:
                self.fiddler_client.add_model(project_id=self.project, dataset_id=self.model, model_id=self.model, model_info=model_info)
            except Exception as e:
                print(f'Error adding model {self.model}: {e}.Fiddler integration will not work.')
                raise e

    @property
    def custom_features(self) -> list:
        """
        Define custom features for the model to automatically enrich the data with.
        Here, we enable the following enrichments:
        - Automatic Embedding generation for prompt and response
        - Text Statistics such as:
            - Automated Readability Index
            - Coleman Liau Index
            - Dale Chall Readability Score
            - Difficult Words
            - Flesch Reading Ease
            - Flesch Kincaid Grade
            - Gunning Fog
            - Linsear Write Formula
        - PII - Personal Identifiable Information
        - Sentiment Analysis

        """
        return [self.fdl.Enrichment(name='Prompt Embedding', enrichment='embedding', columns=[PROMPT]), self.fdl.TextEmbedding(name='Prompt CF', source_column=PROMPT, column='Prompt Embedding'), self.fdl.Enrichment(name='Response Embedding', enrichment='embedding', columns=[RESPONSE]), self.fdl.TextEmbedding(name='Response CF', source_column=RESPONSE, column='Response Embedding'), self.fdl.Enrichment(name='Text Statistics', enrichment='textstat', columns=[PROMPT, RESPONSE], config={'statistics': ['automated_readability_index', 'coleman_liau_index', 'dale_chall_readability_score', 'difficult_words', 'flesch_reading_ease', 'flesch_kincaid_grade', 'gunning_fog', 'linsear_write_formula']}), self.fdl.Enrichment(name='PII', enrichment='pii', columns=[PROMPT, RESPONSE]), self.fdl.Enrichment(name='Sentiment', enrichment='sentiment', columns=[PROMPT, RESPONSE])]

    def _publish_events(self, run_id: UUID, prompt_responses: List[str], duration: int, llm_status: str, model_name: Optional[str]='', token_usage_dict: Optional[Dict[str, Any]]=None) -> None:
        """
        Publish events to fiddler
        """
        prompt_count = len(self.run_id_prompts[run_id])
        df = self.pd.DataFrame({PROMPT: self.run_id_prompts[run_id], RESPONSE: prompt_responses, RUN_ID: [str(run_id)] * prompt_count, DURATION: [duration] * prompt_count, LLM_STATUS: [llm_status] * prompt_count, MODEL_NAME: [model_name] * prompt_count})
        if token_usage_dict:
            for key, value in token_usage_dict.items():
                df[key] = [value] * prompt_count if isinstance(value, int) else value
        try:
            if df.shape[0] > 1:
                self.fiddler_client.publish_events_batch(self.project, self.model, df)
            else:
                df_dict = df.to_dict(orient='records')
                self.fiddler_client.publish_event(self.project, self.model, event=df_dict[0])
        except Exception as e:
            print(f'Error publishing events to fiddler: {e}. continuing...')

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        run_id = kwargs[RUN_ID]
        self.run_id_prompts[run_id] = prompts
        self.run_id_starttime[run_id] = int(time.time() * 1000)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        flattened_llmresult = response.flatten()
        run_id = kwargs[RUN_ID]
        run_duration = int(time.time() * 1000) - self.run_id_starttime[run_id]
        model_name = ''
        token_usage_dict = {}
        if isinstance(response.llm_output, dict):
            token_usage_dict = {k: v for k, v in response.llm_output.items() if k in [TOTAL_TOKENS, PROMPT_TOKENS, COMPLETION_TOKENS]}
            model_name = response.llm_output.get(MODEL_NAME, '')
        prompt_responses = [llmresult.generations[0][0].text for llmresult in flattened_llmresult]
        self._publish_events(run_id, prompt_responses, run_duration, SUCCESS, model_name, token_usage_dict)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs[RUN_ID]
        duration = int(time.time() * 1000) - self.run_id_starttime[run_id]
        self._publish_events(run_id, [''] * len(self.run_id_prompts[run_id]), duration, FAILURE)