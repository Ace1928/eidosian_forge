import os


def create_directories(base_path, tree_structure):
    for path in tree_structure:
        try:
            os.makedirs(os.path.join(base_path, path))
            print(f"Directory created: {path}")
        except FileExistsError:
            print(f"Directory already exists: {path}")


tree_structure = [
    "docs/technical/architecture",
    "docs/technical/design_principles",
    "docs/technical/coding_standards",
    "docs/technical/performance_optimization",
    "docs/technical/scalability_considerations",
    "docs/technical/extensibility_guidelines",
    "docs/technical/module_interactions",
    "docs/tutorials",
    "docs/API",
    "evie/node_tools/logic",
    "evie/node_tools/algorithms",
    "evie/synapse_tools/logic",
    "evie/synapse_tools/algorithms",
    "evie/training_tools",
    "evie/modality_tools",
    "evie/pipeline_tools",
    "evie/layer_tools",
    "evie/weight_tools",
    "evie/data_type_tools",
    "evie/tokenization_tools",
    "evie/processing_tools/text_utils",
    "evie/processing_tools/datetime_utils",
    "evie/processing_tools/file_utils",
    "evie/processing_tools/system_utils",
    "evie/knowledgebase_tools",
    "evie/utils",
    "frontend/templates",
    "frontend/static/css",
    "frontend/static/js",
    "frontend/static/img/icons",
    "backend/api",
    "backend/core",
    "tests/unit",
    "tests/integration",
    "tests/performance",
    "scripts/data_processing",
    "scripts/model_training",
    "scripts/model_evaluation",
    "scripts/model_deployment",
]

if __name__ == "__main__":
    base_path = os.getcwd()  # Current working directory
    create_directories(base_path, tree_structure)
