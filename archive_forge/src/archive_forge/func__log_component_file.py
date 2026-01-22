def _log_component_file(func, run=None):
    name = func.__name__
    output_component_file = f'{name}.yml'
    components._python_op.func_to_component_file(func, output_component_file)
    artifact = wandb.Artifact(name, type='kubeflow_component_file')
    artifact.add_file(output_component_file)
    run.log_artifact(artifact)
    wandb.termlog(f'Logging component file: {output_component_file}')