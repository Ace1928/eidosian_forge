#!/usr/bin/env python3
"""
Eidosian Virtual Environment & Repo Forge Setup
────────────────────────────────────────────────────────────
Fully universal setup script for all *_forge repositories.
This script:
  • Verifies system environment and Python prerequisites.
  • Creates a robust virtual environment (using venv) with no hardcoding.
  • Generates activation helper scripts (bash and fish).
  • Installs base dependencies and installs repositories (in editable mode).
  • Creates symlink wrappers for CLI commands.
  • Runs tests and displays usage instructions.
All output is styled in true Eidosian fashion.
"""

import os
import sys
import re
import json
import subprocess
import shutil
import glob
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

from setup_repo_forge import WHITE

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANSI Terminal Colors – Eidosian Precision in Presentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
PURPLE = "\033[0;35m"
CYAN = "\033[0;36m"
GRAY = "\033[0;90m"
WHITE = "\033[1;37m"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Global Paths and Repository List
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCRIPT_DIR = Path(__file__).resolve().parent
# The primary repo directory for repo_forge (assumed to be under SCRIPT_DIR)
REPO_PATH = SCRIPT_DIR / "repo_forge_repo"
# The virtual environment will be created at this path
VENV_PATH = SCRIPT_DIR / "eidos_venv"
ACTIVATION_SCRIPT = VENV_PATH / "bin" / "activate"
# List of forge repositories (names as directories)
REPOS = [
    "repo_forge_repo",
    "metadata_forge_repo",
    "ollama_forge_repo",
    "terminal_forge_repo",
    "doc_forge_repo",
    "memory_forge_repo",
    "refactor_forge_repo",
    "file_forge_repo",
    "knowledge_forge_repo",
]
# Centralized versions database path (for cross‑repo tracking)
CENTRAL_VERSIONS_PATH = os.getenv("CENTRAL_VERSIONS_PATH", str(SCRIPT_DIR / "central_versions.json"))
ACTIVATE_WRAPPER = SCRIPT_DIR / "activate_eidos_venv.sh"
LOG_FILE = SCRIPT_DIR / "repo_forge_setup.log"

# Execution parameters (defaults)
BOOTSTRAP_MODE = False
FORCE_REINSTALL = False
DEV_DEPENDENCIES = True
SKIP_TESTS = False
DEBUG_MODE = False
TARGET_VERSION = ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logging Setup – Dual Output (Terminal and Log File)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
logging.basicConfig(
    level=logging.DEBUG if os.getenv("FORGE_DEBUG") == "1" or DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("forge.setup")
logger.debug("Eidosian Repo Forge Setup Script loaded with debug logging")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Terminal Logging Helpers – Eidosian Style
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def log_debug(msg):
    logger.debug(msg)

def log_info(msg):
    logger.info(f"{BLUE}[INFO]{RESET} {msg}")

def log_success(msg):
    logger.info(f"{GREEN}[SUCCESS]{RESET} {msg}")

def log_warning(msg):
    logger.warning(f"{YELLOW}[WARNING]{RESET} {msg}")

def log_error(msg):
    logger.error(f"{RED}[ERROR]{RESET} {msg}")

def log_step(msg):
    print(f"\n{PURPLE}[STEP]{RESET} {BOLD}{msg}{RESET}")
    logger.info(f"[STEP] {msg}")

def log_code(msg):
    print(f"  {CYAN}{msg}{RESET}")
    logger.info(f"  {msg}")

def log_header(title):
    width = 72
    padding = (width - len(title) - 4) // 2
    left = " " * padding
    right = " " * (padding + ((width - len(title) - 4) % 2))
    header_line = f"{PURPLE}╔{'═'*width}╗{RESET}"
    title_line = f"{PURPLE}║{left}{WHITE}{BOLD}{title}{RESET}{PURPLE}{right}║{RESET}"
    footer_line = f"{PURPLE}╚{'═'*width}╝{RESET}"
    print(f"\n{header_line}\n{title_line}\n{footer_line}\n")
    logger.info(header_line)
    logger.info(title_line)
    logger.info(footer_line)

def log_execution_time(start_time):
    elapsed = time.time() - start_time
    formatted = f"{elapsed:.3f}"
    log_debug(f"Execution time: {formatted}s")
    print(f"{GRAY}⏱️ Execution time: {formatted}s{RESET}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Environment Verification Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def verify_python():
    log_step("Verifying Python installation...")
    if shutil.which("python3") is None:
        log_error("Python 3 not found. Please install Python 3.8 or higher.")
        sys.exit(1)
    # Get Python version
    try:
        version_output = subprocess.check_output(["python3", "--version"]).decode().strip()
    except Exception:
        version_output = "Unknown"
    log_debug(f"Detected: {version_output}")
    major, minor, *_ = version_output.split()[1].split('.')
    if int(major) < 3 or (int(major) == 3 and int(minor) < 8):
        log_error(f"Python 3.8+ is required (detected {version_output})")
        sys.exit(1)
    log_success(f"Using {version_output}")

def verify_venv_module():
    log_step("Verifying venv module availability...")
    try:
        subprocess.run(["python3", "-m", "venv", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        log_success("Python venv module is available")
    except subprocess.CalledProcessError:
        log_error("Python venv module is not available. Please install it (e.g. sudo apt install python3-venv)")
        sys.exit(1)

def verify_pip():
    log_step("Verifying pip functionality...")
    try:
        subprocess.run(["python3", "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        log_success("pip is available")
    except subprocess.CalledProcessError:
        log_warning("pip may not be installed correctly. It will be upgraded during environment setup.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Virtual Environment Setup Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def create_venv(force=False):
    if VENV_PATH.exists():
        log_warning(f"Virtual environment already exists at {VENV_PATH}")
        if force:
            log_step("Removing existing virtual environment...")
            shutil.rmtree(VENV_PATH)
        else:
            return
    log_step(f"Creating Eidosian virtual environment at {VENV_PATH}")
    try:
        subprocess.run(["python3", "-m", "venv", str(VENV_PATH), "--prompt=🔮 eidos"], check=True)
        log_success("Virtual environment created successfully")
    except subprocess.CalledProcessError:
        log_error("Failed to create virtual environment")
        sys.exit(1)

def activate_venv():
    log_step("Activating Eidosian virtual environment...")
    if ACTIVATION_SCRIPT.exists():
        # Note: We cannot truly 'source' a script in Python.
        # Instead, adjust PATH and VIRTUAL_ENV environment variables.
        bin_path = str(VENV_PATH / "bin")
        os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
        os.environ["VIRTUAL_ENV"] = str(VENV_PATH)
        log_success("Virtual environment activated (environment variables adjusted)")
    else:
        log_error(f"Activation script not found at {ACTIVATION_SCRIPT}")
        sys.exit(1)

def create_activation_helpers():
    log_step("Creating activation helper scripts...")
    # Create bash activation helper
    bash_helper = SCRIPT_DIR / "activate_eidos_venv.sh"
    bash_content = f"""#!/bin/bash
# Activate the Eidosian virtual environment
VENV_PATH="{VENV_PATH}"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo -e "\\033[0;35m[EIDOS]\\033[0m Activating Eidosian virtual environment..."
    source "$VENV_PATH/bin/activate"
    # Optionally add forge repos to PYTHONPATH
    for repo in {" ".join(REPOS)}; do
        repo_path="$(dirname "$VENV_PATH")/$repo"
        if [ -d "$repo_path" ]; then
            export PYTHONPATH="$repo_path:$PYTHONPATH"
        fi
    done
    echo -e "\\033[0;32m[SUCCESS]\\033[0m Environment activated."
else
    echo -e "\\033[0;31m[ERROR]\\033[0m Virtual environment not found at $VENV_PATH"
fi
"""
    bash_helper.write_text(bash_content)
    os.chmod(bash_helper, 0o755)
    log_success(f"Bash helper created at {bash_helper}")

    # Create fish shell activation helper
    fish_helper_dir = VENV_PATH / "fish"
    fish_helper_dir.mkdir(parents=True, exist_ok=True)
    fish_helper = fish_helper_dir / "activate.fish"
    fish_content = f"""function _eidos_venv_prompt
    echo -n "🔮 (eidos) "
end

if status is-interactive
    set -g __old_prompt $fish_prompt
    for repo in {" ".join(REPOS)}
        set repo_path "{SCRIPT_DIR}/$repo"
        if test -d $repo_path
            set -x PYTHONPATH $repo_path $PYTHONPATH
        end
    end
    functions -c fish_prompt __old_fish_prompt
    function fish_prompt
        _eidos_venv_prompt; __old_fish_prompt
    end
end

function deactivate
    if functions -q __old_fish_prompt
        functions -e fish_prompt
        functions -c __old_fish_prompt fish_prompt
        functions -e __old_fish_prompt
    end
    echo "Deactivated Eidosian virtual environment"
end
"""
    fish_helper.write_text(fish_content)
    log_success(f"Fish helper created at {fish_helper}")

    # Create .pth file for additional repos
    for site_path in glob.glob(str(VENV_PATH / "lib/python3.*" / "site-packages")):
        pth_file = Path(site_path) / "eidos_repos.pth"
        with open(pth_file, "w", encoding="utf-8") as f:
            f.write("# Eidosian repository paths for seamless imports\n")
            for repo in REPOS:
                repo_dir = SCRIPT_DIR / repo
                if repo_dir.exists():
                    f.write(str(repo_dir) + "\n")
        log_success(f"Created {pth_file}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Base Package Installation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def install_base_packages():
    log_step("Installing/upgrading base packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
        # Install common packages
        packages = ["colorama", "pyyaml", "jinja2", "pytest", "black", "sphinx", "furo", "myst-parser"]
        subprocess.run([sys.executable, "-m", "pip", "install"] + packages, check=True)
        log_success("Base packages installed/upgraded")
    except subprocess.CalledProcessError:
        log_error("Failed to install base packages")
        sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Repository Installation Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def install_repo_packages():
    log_step("Installing repositories in development mode...")
    # First, install repo_forge_repo (the core repo)
    core_repo = SCRIPT_DIR / "repo_forge_repo"
    if core_repo.exists():
        log_info("Installing repo_forge_repo...")
        os.chdir(core_repo)
        if (core_repo / "setup.py").exists() or (core_repo / "pyproject.toml").exists():
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
                log_success("repo_forge_repo installed successfully")
                # Attempt to install dev and docs extras if present
                if (core_repo / "pyproject.toml").exists():
                    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], check=False)
                    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[docs]"], check=False)
            except subprocess.CalledProcessError:
                log_error("Failed to install repo_forge_repo")
                sys.exit(1)
        else:
            log_warning("No setup.py or pyproject.toml found in repo_forge_repo")
    else:
        log_error("repo_forge_repo not found! Cannot continue with other repositories.")
        sys.exit(1)
    
    # Install all other repositories
    for repo in REPOS:
        if repo == "repo_forge_repo":
            continue
        repo_dir = SCRIPT_DIR / repo
        if repo_dir.exists():
            log_info(f"Installing {repo} in development mode...")
            os.chdir(repo_dir)
            if (repo_dir / "setup.py").exists() or (repo_dir / "pyproject.toml").exists():
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
                    log_success(f"{repo} installed successfully")
                    if (repo_dir / "pyproject.toml").exists():
                        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"], check=False)
                        subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[docs]"], check=False)
                except subprocess.CalledProcessError:
                    log_warning(f"Failed to install {repo} in development mode")
            else:
                log_warning(f"No setup.py or pyproject.toml found for {repo}")
        else:
            log_warning(f"Repository {repo} not found at {repo_dir}")
    
    # Install all dependencies together
    log_info("Installing all common dependencies...")
    common_packages = ["colorama", "pyyaml", "jinja2", "pytest", "black", "sphinx", "furo", "myst-parser"]
    subprocess.run([sys.executable, "-m", "pip", "install"] + common_packages, check=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Executable Symlink Creation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def create_symlink_executables():
    log_step("Creating executable symlinks...")
    local_bin = VENV_PATH / "local" / "bin"
    local_bin.mkdir(parents=True, exist_ok=True)
    
    target = VENV_PATH / "bin" / "repo-forge"
    if target.exists():
        symlink = VENV_PATH / "local" / "bin" / "repo-forge"
        if symlink.exists():
            symlink.unlink()
        os.symlink(target, symlink)
        log_success("Created symlink for repo-forge")
    else:
        log_warning("repo-forge command not found in venv/bin")
    
    # (Optionally, add additional symlinks for other forge commands)
    
    # Ensure local/bin is added to PATH in the activation script
    activate_file = VENV_PATH / "bin" / "activate"
    if activate_file.exists():
        with open(activate_file, "r", encoding="utf-8") as f:
            content = f.read()
        if "local/bin" not in content:
            new_content = content.replace('PATH="$VIRTUAL_ENV/bin:$PATH"', 'PATH="$VIRTUAL_ENV/bin:$VIRTUAL_ENV/local/bin:$PATH"')
            with open(activate_file, "w", encoding="utf-8") as f:
                f.write(new_content)
            log_success("Updated activation script to include local/bin in PATH")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Verification & Testing Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def verify_installation():
    log_step("Verifying installations...")
    try:
        subprocess.run([sys.executable, "-c", "import sys; print(f'Python {sys.version}'); print(f'Executable: {sys.executable}')"], check=True)
        log_success("Python environment verified")
    except subprocess.CalledProcessError:
        log_error("Failed to verify Python environment")
    
    # Check repo_forge import
    try:
        subprocess.run([sys.executable, "-c", "import repo_forge; print(f'repo_forge {repo_forge.__version__}')"], check=True)
        log_success("repo_forge imported successfully")
    except subprocess.CalledProcessError:
        log_warning("repo_forge not importable (may not be created yet)")
    
    # Check metadata_forge if available
    try:
        subprocess.run([sys.executable, "-c", "import metadata_forge; print(f'metadata_forge {metadata_forge.__version__}')"], check=True)
        log_success("metadata_forge imported successfully")
    except subprocess.CalledProcessError:
        log_warning("metadata_forge not importable (if not created yet)")
    
    # Check essential packages
    try:
        code = "import colorama, pytest, wheel"
        subprocess.run([sys.executable, "-c", code], check=True)
        log_success("Essential packages verified")
    except subprocess.CalledProcessError:
        log_warning("Some essential packages may not be installed correctly")

def run_tests():
    log_step("Running verification tests...")
    if SKIP_TESTS:
        log_info("Tests skipped due to --skip-tests flag")
        return
    os.chdir(REPO_PATH)
    try:
        subprocess.run([sys.executable, "-m", "pytest", "-xvs", "tests/"], check=True)
        log_success("All tests passed")
    except subprocess.CalledProcessError:
        log_warning("Some tests failed")
    # Test CLI functionality
    try:
        subprocess.run(["repo-forge", "--version"], check=True)
        log_success("CLI test passed")
    except subprocess.CalledProcessError:
        log_warning("repo-forge command not found; trying module invocation")
        try:
            subprocess.run([sys.executable, "-m", "repo_forge.cli", "--version"], check=True)
            log_success("Module CLI test passed")
        except subprocess.CalledProcessError:
            log_error("CLI test failed")
            sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Usage Instructions Display
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def print_usage_instructions():
    print("")
    log_header("EIDOSIAN VIRTUAL ENVIRONMENT USAGE INSTRUCTIONS")
    print("")
    log_info("To activate the virtual environment:")
    if Path(ACTIVATE_WRAPPER).exists():
        log_code(f"source {ACTIVATE_WRAPPER}")
    else:
        log_code(f"source {ACTIVATION_SCRIPT}")
    print("")
    log_info("To deactivate:")
    log_code("deactivate")
    print("")
    log_info("To set up repo_forge:")
    log_code(f"python {SCRIPT_DIR / 'setup_repo_forge.py'}")
    print("")
    log_info("For metadata repository setup:")
    log_code(f"python {SCRIPT_DIR / 'setup_metadata_forge.py'}")
    print("")
    log_info("Available commands after setup:")
    log_code("repo-forge --help")
    log_code("metadata-forge --help")
    print("")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command-Line Argument Parsing – Eidosian Interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_arguments():
    parser = argparse.ArgumentParser(description="Eidosian Repo Forge Setup Script")
    parser.add_argument("--force", "-f", action="store_true", help="Force reinstall/recreation of virtual environment")
    parser.add_argument("--bootstrap", "-b", action="store_true", help="Run full repository bootstrapping")
    parser.add_argument("--skip-dev", "-s", action="store_true", help="Skip installing development dependencies")
    parser.add_argument("--skip-tests", "-t", action="store_true", help="Skip running tests")
    parser.add_argument("--debug", "-d", action="store_true", help="Show debug output")
    parser.add_argument("--version", "-v", metavar="VER", help="Install specific version (if applicable)")
    return parser.parse_args()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Execution Sequence – Eidosian Orchestration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    global FORCE_REINSTALL, BOOTSTRAP_MODE, DEV_DEPENDENCIES, SKIP_TESTS, DEBUG_MODE, TARGET_VERSION
    args = parse_arguments()
    FORCE_REINSTALL = args.force
    BOOTSTRAP_MODE = args.bootstrap
    DEV_DEPENDENCIES = not args.skip_dev
    SKIP_TESTS = args.skip_tests
    DEBUG_MODE = args.debug
    if args.version:
        TARGET_VERSION = args.version

    total_start = time.time()

    # Initialize log file header
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("╔" + "═"*60 + "╗\n")
        f.write(f"║  REPO FORGE SETUP LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ║\n")
        f.write("╚" + "═"*60 + "╝\n\n")
    log_debug("Logging initialized")

    log_header("REPO FORGE SETUP SEQUENCE")
    log_info("Beginning setup with mathematical precision and Eidosian elegance")

    try:
        verify_python()
        verify_venv_module()
        verify_pip()
        create_venv(force=FORCE_REINSTALL)
        activate_venv()
        create_activation_helpers()
        install_base_packages()
        install_repo_packages()
        create_symlink_executables()
        verify_installation()
        # If bootstrapping is enabled, run the bootstrap command
        if BOOTSTRAP_MODE:
            log_step("Bootstrapping repository structure...")
            # Run the bootstrap via the repo_forge.cli module
            subprocess.run([sys.executable, "-m", "repo_forge.cli", "--bootstrap", "--verbose"], check=True)
        else:
            log_info("Minimal bootstrap mode enabled")
            subprocess.run([sys.executable, "-m", "repo_forge.cli", "--bootstrap"], check=True)
        run_tests()
    except Exception as e:
        log_error(f"Setup failed: {e}")
        sys.exit(1)

    total_elapsed = time.time() - total_start
    log_header("SETUP COMPLETED SUCCESSFULLY")
    log_success(f"Total execution time: {total_elapsed:.3f} seconds")
    print_usage_instructions()
    return 0

if __name__ == "__main__":
    sys.exit(main())
