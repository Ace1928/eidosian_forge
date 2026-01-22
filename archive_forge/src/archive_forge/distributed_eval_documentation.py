import parlai.scripts.eval_model as eval_model
import parlai.utils.distributed as distributed_utils

Distributed evaluation script. NOT MEANT TO BE CALLED DIRECTLY BY USER.

This script is meant to be in conjunction with
`SLURM <https://slurm.schedmd.com/>`, which provides environmental variables
describing the environment.

An example sbatch script is below, for a 2-host, 8-GPU setup (16 total gpus):

.. code-block:: bash



  #!/bin/sh
  #SBATCH --job-name=distributed_example
  #SBATCH --output=/path/to/savepoint/stdout.%j
  #SBATCH --error=/path/to/savepoint/stderr.%j
  #SBATCH --partition=priority
  #SBATCH --nodes=2
  #SBATCH --time=0:10:00
  #SBATCH --signal=SIGINT
  #SBATCH --gres=gpu:8
  #SBATCH --ntasks-per-node=8
  #SBATCH --mem=64G
  #SBATCH --cpus-per-task=10
  srun python -u -m parlai.scripts.distributed_eval     -m seq2seq -t convai2 --dict-file /path/to/dict-file
