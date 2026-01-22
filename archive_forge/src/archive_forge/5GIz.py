import os


def add_dotenv_loader(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".py"):
                filepath = os.path.join(subdir, filename)
                with open(filepath, "r+") as f:
                    content = f.read()
                    f.seek(0, 0)
                    f.write(
                        "import sys\nfrom pathlib import Path\n\n# Add the current directory to sys.path\ncurrent_directory = Path(__file__).parent.absolute()\nsys.path.append(str(current_directory))\n\n"
                        + content
                    )


root_dir = "/home/lloyd/Dropbox/evie_env/cursor/RWKV-Runner/backend-python"
add_dotenv_loader(root_dir)
