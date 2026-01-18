import logging
import sys
from collections import defaultdict
from typing import (
from uuid import UUID
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.outputs import LLMResult
def uptrain_evaluate(self, project_name: str, data: List[Dict[str, Any]], checks: List[str]) -> None:
    """Run an evaluation on the UpTrain server using UpTrain client."""
    if self.uptrain_client.__class__.__name__ == 'APIClient':
        uptrain_result = self.uptrain_client.log_and_evaluate(project_name=project_name, data=data, checks=checks)
    else:
        uptrain_result = self.uptrain_client.evaluate(data=data, checks=checks)
    self.schema.uptrain_results[project_name].append(uptrain_result)
    score_name_map = {'score_context_relevance': 'Context Relevance Score', 'score_factual_accuracy': 'Factual Accuracy Score', 'score_response_completeness': 'Response Completeness Score', 'score_sub_query_completeness': 'Sub Query Completeness Score', 'score_context_reranking': 'Context Reranking Score', 'score_context_conciseness': 'Context Conciseness Score', 'score_multi_query_accuracy': 'Multi Query Accuracy Score'}
    if self.log_results:
        logger.setLevel(logging.INFO)
    for row in uptrain_result:
        columns = list(row.keys())
        for column in columns:
            if column == 'question':
                logger.info(f'\nQuestion: {row[column]}')
                self.first_score_printed_flag = False
            elif column == 'response':
                logger.info(f'Response: {row[column]}')
                self.first_score_printed_flag = False
            elif column == 'variants':
                logger.info('Multi Queries:')
                for variant in row[column]:
                    logger.info(f'  - {variant}')
                self.first_score_printed_flag = False
            elif column.startswith('score'):
                if not self.first_score_printed_flag:
                    logger.info('')
                    self.first_score_printed_flag = True
                if column in score_name_map:
                    logger.info(f'{score_name_map[column]}: {row[column]}')
                else:
                    logger.info(f'{column}: {row[column]}')
    if self.log_results:
        logger.setLevel(logging.WARNING)