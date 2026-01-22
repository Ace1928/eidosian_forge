from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict, Union
from langsmith.evaluation.evaluator import run_evaluator
from langsmith.run_helpers import traceable
from langsmith.schemas import Example, Run
def as_run_evaluator(self) -> RunEvaluator:
    """Convert the LangChainStringEvaluator to a RunEvaluator.

        This is the object used in the LangSmith `evaluate` API.

        Returns:
            RunEvaluator: The converted RunEvaluator.
        """
    input_str = '\n       "input": example.inputs[\'input\'],' if self.evaluator.requires_input else ''
    reference_str = '\n       "reference": example.outputs[\'expected\']' if self.evaluator.requires_reference else ''
    customization_error_str = f"""\ndef prepare_data(run, example):\n    return {{\n        "prediction": run.outputs['my_output'],{reference_str}{input_str}\n    }}\nevaluator = LangChainStringEvaluator(..., prepare_data=prepare_data)\n"""

    @traceable
    def prepare_evaluator_inputs(run: Run, example: Optional[Example]=None) -> SingleEvaluatorInput:
        if run.outputs and len(run.outputs) > 1:
            raise ValueError(f'Evaluator {self.evaluator} only supports a single prediction key. Please ensure that the run has a single output. Or initialize with a prepare_data:\n{customization_error_str}')
        if self.evaluator.requires_reference and example and example.outputs and (len(example.outputs) > 1):
            raise ValueError(f'Evaluator {self.evaluator} nly supports a single reference key. Please ensure that the example has a single output. Or create a custom evaluator yourself:\n{customization_error_str}')
        if self.evaluator.requires_input and example and example.inputs and (len(example.inputs) > 1):
            raise ValueError(f'Evaluator {self.evaluator} only supports a single input key. Please ensure that the example has a single input. Or initialize with a prepare_data:\n{customization_error_str}')
        return SingleEvaluatorInput(prediction=next(iter(run.outputs.values())), reference=next(iter(example.outputs.values())) if self.evaluator.requires_reference and example and example.outputs else None, input=next(iter(example.inputs.values())) if self.evaluator.requires_input and example and example.inputs else None)

    @traceable(name=self.evaluator.evaluation_name)
    def evaluate(run: Run, example: Optional[Example]=None) -> dict:
        eval_inputs = prepare_evaluator_inputs(run, example) if self._prepare_data is None else self._prepare_data(run, example)
        results = self.evaluator.evaluate_strings(**eval_inputs)
        return {'key': self.evaluator.evaluation_name, **results}
    return run_evaluator(evaluate)