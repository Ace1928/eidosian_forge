from pathlib import Path
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython.utils.tempdir import TemporaryWorkingDirectory
def test_temporary_working_directory():
    with TemporaryWorkingDirectory() as directory:
        directory_path = Path(directory).resolve()
        assert directory_path.exists()
        assert Path.cwd().resolve() == directory_path
    assert not directory_path.exists()
    assert Path.cwd().resolve() != directory_path