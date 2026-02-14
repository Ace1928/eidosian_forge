# Development Archaeology Curation Report

- Timestamp: 2026-02-14T17:32:46+10:00
- Development HEAD: 3abb67f
- Forge root: /home/lloyd/eidosian_forge

## Coverage Matrix

| Source | Files | Suggested Destination | Status |
|---|---:|---|---|
| ECosmos | 9 | `game_forge/src/ECosmos` | present |
| chess_game | 12 | `game_forge/src/chess_game` | present |
| EMemory | 17 | `memory_forge/legacy_imports/development_head_2026-02-14_174300/EMemory` | present |
| python_repository | 72 | `llm_forge/legacy_imports/development_head_2026-02-14_174300/python_repository` | present |
| oumi-main | 294 | `llm_forge/legacy_imports/development_head_2026-02-14_174300/oumi-main` | present |
| eidos_framework | 5 | `llm_forge/legacy_imports/development_head_2026-02-14_174300/eidos_framework` | present |
| templates | 8 | `prompt_forge/legacy_imports/development_head_2026-02-14_174300/templates` | present |
| notebooks | 18 | `doc_forge/legacy_imports/development_head_2026-02-14_174300/notebooks` | present |
| papers | 12 | `doc_forge/legacy_imports/development_head_2026-02-14_174300/papers` | present |

## Canonical Drift Check

### ECosmos
- Only in Forge/game_forge/src/ECosmos: __init__.py
- Only in Forge/game_forge/src/ECosmos: __main__.py
- Only in Forge/game_forge/src/ECosmos: __pycache__
- Files Development_HEAD/ECosmos/config.py and Forge/game_forge/src/ECosmos/config.py differ
- Files Development_HEAD/ECosmos/data_structures.py and Forge/game_forge/src/ECosmos/data_structures.py differ
- Only in Forge/game_forge/src/ECosmos: ecosystem.log
- Files Development_HEAD/ECosmos/evolution.py and Forge/game_forge/src/ECosmos/evolution.py differ
- Files Development_HEAD/ECosmos/instructions.py and Forge/game_forge/src/ECosmos/instructions.py differ
- Files Development_HEAD/ECosmos/interpreter.py and Forge/game_forge/src/ECosmos/interpreter.py differ
- Files Development_HEAD/ECosmos/main.py and Forge/game_forge/src/ECosmos/main.py differ
- Files Development_HEAD/ECosmos/mvp.py and Forge/game_forge/src/ECosmos/mvp.py differ
- Only in Forge/game_forge/src/ECosmos: state
- Files Development_HEAD/ECosmos/state_manager.py and Forge/game_forge/src/ECosmos/state_manager.py differ
- Files Development_HEAD/ECosmos/visualize.py and Forge/game_forge/src/ECosmos/visualize.py differ

### chess_game
- Only in Forge/game_forge/src/chess_game: __init__.py
- Only in Forge/game_forge/src/chess_game: __main__.py
- Only in Forge/game_forge/src/chess_game: __pycache__
- Files Development_HEAD/chess_game/color_mapper.py and Forge/game_forge/src/chess_game/color_mapper.py differ
- Files Development_HEAD/chess_game/local_model_network.py and Forge/game_forge/src/chess_game/local_model_network.py differ
- Files Development_HEAD/chess_game/modular_local_model_network.py and Forge/game_forge/src/chess_game/modular_local_model_network.py differ
- Files Development_HEAD/chess_game/piecerenderer.py and Forge/game_forge/src/chess_game/piecerenderer.py differ
- Files Development_HEAD/chess_game/prototype.py and Forge/game_forge/src/chess_game/prototype.py differ
- Files Development_HEAD/chess_game/prototype_old.py and Forge/game_forge/src/chess_game/prototype_old.py differ
- Files Development_HEAD/chess_game/textprocessor.py and Forge/game_forge/src/chess_game/textprocessor.py differ
- Files Development_HEAD/chess_game/usermanager.py and Forge/game_forge/src/chess_game/usermanager.py differ

## Recommended Next Upgrades

1. Create a deterministic merge pass for ECosmos/chess_game (3-way merge against Development HEAD).
2. Add smoke tests for imported EMemory and python_repository entry points.
3. Promote stable notebooks/papers into doc_forge index manifests with topic tags.
4. Keep raw legacy imports immutable under timestamped directories; build curated derivatives in forge-native paths.
