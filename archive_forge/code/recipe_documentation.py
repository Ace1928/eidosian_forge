import logging
from typing import Any, Optional
from mlflow.recipes.recipe import BaseRecipe
from mlflow.recipes.step import BaseStep
from mlflow.recipes.steps.evaluate import EvaluateStep
from mlflow.recipes.steps.ingest import IngestScoringStep, IngestStep
from mlflow.recipes.steps.predict import PredictStep
from mlflow.recipes.steps.register import RegisterStep
from mlflow.recipes.steps.split import SplitStep
from mlflow.recipes.steps.train import TrainStep
from mlflow.recipes.steps.transform import TransformStep

        Displays a visual overview of the recipe graph, or displays a summary of results from
        a particular recipe step if specified. If the specified step has not been executed,
        nothing is displayed.

        Args:
            step: String name of the recipe step for which to display a results summary. If
                unspecified, a visual overview of the recipe graph is displayed.

            .. code-block:: python
              import os
              from mlflow.recipes import Recipe

              os.chdir("~/recipes-regression-template")
              regression_recipe = Recipe(profile="local")
              # Display a visual overview of the recipe graph.
              regression_recipe.inspect()
              # Run the 'train' recipe step
              regression_recipe.run(step="train")
              # Display a summary of results from the preceding 'transform' step
              regression_recipe.inspect(step="transform")
        