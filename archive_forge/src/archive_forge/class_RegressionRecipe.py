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
class RegressionRecipe(BaseRecipe):
    """
    A recipe for developing high-quality regression models. The recipe is designed for
    developing models using scikit-learn and frameworks that integrate with scikit-learn,
    such as the ``XGBRegressor`` API from XGBoost. The corresponding recipe
    template repository is available at https://github.com/mlflow/recipes-regression-template.
    The training recipe contains the following sequential steps:

    **ingest** -> **split** -> **transform** -> **train** -> **evaluate** -> **register**

    while the batch scoring recipe contains this set of sequential steps:

    **ingest_scoring** -> **predict**

    .. code-block:: python
        :caption: Example

        import os
        from mlflow.recipes import Recipe

        os.chdir("~/recipes-regression-template")
        regression_recipe = Recipe(profile="local")
        # Display a visual overview of the recipe graph
        regression_recipe.inspect()
        # Run the full recipe
        regression_recipe.run()
        # Display a summary of results from the 'train' step, including the trained model
        # and associated performance metrics computed from the training & validation datasets
        regression_recipe.inspect(step="train")
        # Display a summary of results from the 'evaluate' step, including model explanations
        # computed from the validation dataset and metrics computed from the test dataset
        regression_recipe.inspect(step="evaluate")
    """
    _RECIPE_STEPS = (IngestStep, SplitStep, TransformStep, TrainStep, EvaluateStep, RegisterStep, IngestScoringStep, PredictStep)
    _DEFAULT_STEP_INDEX = _RECIPE_STEPS.index(RegisterStep)

    def _get_step_classes(self):
        return self._RECIPE_STEPS

    def _get_default_step(self) -> BaseStep:
        return self._steps[self._DEFAULT_STEP_INDEX]

    def run(self, step: Optional[str]=None) -> None:
        """
        Runs the full recipe or a particular recipe step, producing outputs and displaying a
        summary of results upon completion. Step outputs are cached from previous executions, and
        steps are only re-executed if configuration or code changes have been made to the step or
        to any of its dependent steps (e.g. changes to the recipe's ``recipe.yaml`` file or
        ``steps/ingest.py`` file) since the previous execution.

        Args:
            step: String name of the step to run within the regression recipe. The step and
                its dependencies are executed sequentially. If a step is not specified, the
                entire recipe is executed. Supported steps, in their order of execution, are:

                - ``"ingest"``: resolves the dataset specified by the ``data/training`` section
                  in the recipe's configuration file (``recipe.yaml``) and converts it to
                  parquet format.

                - ``"ingest_scoring"``: resolves the dataset specified by the
                  ``ingest_scoring`` section in the recipe's configuration file
                  (``recipe.yaml``)and converts it to parquet format.

                - ``"split"``: splits the ingested dataset produced by the **ingest** step into
                  a training dataset for model training, a validation dataset for model
                  performance evaluation & tuning, and a test dataset for model performance
                  evaluation.

                - ``"transform"``: uses the training dataset created by the **split** step to
                  fit a transformer that performs the transformations defined in the
                  recipe's ``steps/transform.py`` file. Then, applies the transformer to the
                  training dataset and the validation dataset, creating transformed datasets
                  that are used by subsequent steps for estimator training and model
                  performance evaluation.

                - ``"train"``: uses the transformed training dataset output from the
                  **transform** step to fit an estimator with the type and parameters defined
                  in in the recipe's ``steps/train.py`` file. Then, joins the estimator with
                  the fitted transformer output from the **transform** step to create a model
                  recipe. Finally, evaluates the model recipe against the transformed
                  training and validation datasets to compute performance metrics.

                - ``"evaluate"``: evaluates the model recipe created by the **train** step
                  on the validation and test dataset outputs from the **split** step, computing
                  performance metrics and model explanations. Then, compares performance
                  metrics against thresholds configured in the recipe's ``recipe.yaml``
                  configuration file to compute a ``model_validation_status``, which indicates
                  whether or not the model is good enough to be registered to the MLflow Model
                  Registry by the subsequent **register** step.

                - ``"register"``: checks the ``model_validation_status`` output of the
                  preceding **evaluate** step and, if model validation was successful (as
                  indicated by the ``'VALIDATED'`` status), registers the model recipe
                  created by the **train** step to the MLflow Model Registry.

                - ``"predict"``: uses the ingested dataset for scoring created by the
                  **ingest_scoring** step and applies the specified model to the dataset.

        .. code-block:: python
            :caption: Example

            import os
            from mlflow.recipes import Recipe

            os.chdir("~/recipes-regression-template")
            regression_recipe = Recipe(profile="local")
            # Run the 'train' step and preceding steps
            regression_recipe.run(step="train")
            # Run the 'register' step and preceding steps; the 'train' step and all steps
            # prior to 'train' are not re-executed because their outputs are already cached
            regression_recipe.run(step="register")
            # Run all recipe steps; equivalent to running 'register'; no steps are re-executed
            # because the outputs of all steps are already cached
            regression_recipe.run()
        """
        return super().run(step=step)

    def get_artifact(self, artifact_name: str) -> Optional[Any]:
        """
        Reads an artifact from the recipe's outputs. Supported artifact names can be obtained by
        examining the recipe graph visualization displayed by
        :py:func:`RegressionRecipe.inspect()`.

        Args:
            artifact_name: The string name of the artifact. Supported artifact values are:

                - ``"ingested_data"``: returns the ingested dataset created in the
                  **ingest** step as a pandas DataFrame.

                - ``"training_data"``: returns the training dataset created in the
                  **split** step as a pandas DataFrame.

                - ``"validation_data"``: returns the validation dataset created in the
                  **split** step as a pandas DataFrame.

                - ``"test_data"``: returns the test dataset created in the **split** step
                  as a pandas DataFrame.

                - ``"ingested_scoring_data"``: returns the scoring dataset created in the
                  **ingest_scoring** step as a pandas DataFrame.

                - ``"transformed_training_data"``: returns the transformed training dataset
                  created in the **transform** step as a pandas DataFrame.

                - ``"transformed_validation_data"``: returns the transformed validation
                  dataset created in the **transform** step as a pandas DataFrame.

                - ``"model"``: returns the MLflow Model recipe created in the **train**
                  step as a :py:class:`PyFuncModel <mlflow.pyfunc.PyFuncModel>` instance.

                - ``"transformer"``: returns the scikit-learn transformer created in the
                  **transform** step.

                - ``"run"``: returns the
                  :py:class:`MLflow Tracking Run <mlflow.entities.Run>` containing the
                  model recipe created in the **train** step and its associated
                  parameters, as well as performance metrics and model explanations created
                  during the **train** and **evaluate** steps.

                - ``"registered_model_version``": returns the MLflow Model Registry
                  :py:class:`ModelVersion <mlflow.entities.model_registry.ModelVersion>`
                  created by the **register** step.

                - ``"scored_data"``: returns the scored dataset created in the
                  **predict** step as a pandas DataFrame.

        Returns:
            An object representation of the artifact corresponding to the specified name,
            as described in the ``artifact_name`` parameter docstring. If the artifact is
            not present because its corresponding step has not been executed or its output
            cache has been cleaned, ``None`` is returned.
        """
        return super().get_artifact(artifact_name=artifact_name)

    def clean(self, step: Optional[str]=None) -> None:
        """
        Removes all recipe outputs from the cache, or removes the cached outputs of a particular
        recipe step if specified. After cached outputs are cleaned for a particular step, the
        step will be re-executed in its entirety the next time it is run.

        Args:
            step: String name of the step to clean within the recipe. If not specified,
                cached outputs are removed for all recipe steps.

        .. code-block:: python

            import os
            from mlflow.recipes import Recipe

            os.chdir("~/recipes-regression-template")
            regression_recipe = Recipe(profile="local")
            # Run the 'train' step and preceding steps
            regression_recipe.run(step="train")
            # Clean the cache of the 'transform' step
            regression_recipe.clean(step="transform")
            # Run the 'split' step; outputs are still cached because 'split' precedes
            # 'transform' & 'train'
            regression_recipe.run(step="split")
            # Run the 'train' step again; the 'transform' and 'train' steps are re-executed because:
            # 1. the cache of the preceding 'transform' step was cleaned and 2. 'train' occurs after
            # 'transform'. The 'ingest' and 'split' steps are not re-executed because their outputs
            # are still cached
            regression_recipe.run(step="train")
        """
        super().clean(step=step)

    def inspect(self, step: Optional[str]=None) -> None:
        """
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
        """
        super().inspect(step=step)